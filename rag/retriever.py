from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.config import Settings
from rag.context_builder import build_context
from rag.documents import DocumentChunk, load_document_chunks
from rag.embeddings import EmbeddingGenerator
from rag.llm import LLMClient
from rag.local_search import LocalSearchEngine
from rag.vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGResult:
    answer: str
    sources: list[dict]
    mode: str


class RAGRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.startup_warning: str | None = None
        self.chunks: list[DocumentChunk] = []
        self.local_engine = LocalSearchEngine([])
        self.embedding_generator: EmbeddingGenerator | None = None
        self.llm_client: LLMClient | None = None
        self.vector_store: VectorStore | None = None
        self.mode = "local"

        self._load_documents()
        self._configure_runtime()

    @property
    def mode_label(self) -> str:
        if self.mode == "rag":
            return "完整 RAG 模式"
        return "本地兜底模式"

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def vector_count(self) -> int:
        if not self.vector_store:
            return 0
        return self.vector_store.count()

    @property
    def components(self) -> dict:
        return {
            "documents": self.chunk_count > 0,
            "embedding_model": self.settings.has_embedding_config,
            "vector_database": self.vector_store is not None,
            "retriever": self.mode == "rag",
            "context_builder": True,
            "llm": self.settings.has_llm_config,
        }

    def answer(self, query: str) -> RAGResult:
        if self.mode == "rag":
            try:
                return self._answer_with_rag(query)
            except Exception as exc:
                LOGGER.exception("Full RAG pipeline failed; falling back to local mode.")
                answer, sources = self.local_engine.answer(query, self.settings.top_k)
                answer = (
                    "完整 RAG 链路调用失败，已临时切换到本地关键词检索兜底模式。\n\n"
                    f"错误摘要：{exc}\n\n{answer}"
                )
                return RAGResult(answer=answer, sources=sources, mode="local")

        answer, sources = self.local_engine.answer(query, self.settings.top_k)
        return RAGResult(answer=answer, sources=sources, mode="local")

    def rebuild_index(self) -> None:
        self._load_documents()
        if self.mode == "rag":
            self._rebuild_vector_index()

    def _load_documents(self) -> None:
        self.chunks = load_document_chunks(
            self.settings.documents_dir,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        self.local_engine = LocalSearchEngine(self.chunks)

    def _configure_runtime(self) -> None:
        if not self.settings.has_rag_config:
            missing = ", ".join(self.settings.missing_rag_config)
            self.startup_warning = (
                f"未完整配置 {missing}，应用正在使用本地关键词检索兜底模式。"
            )
            return

        try:
            self.embedding_generator = EmbeddingGenerator(
                api_key=self.settings.embedding_api_key or "",
                base_url=self.settings.embedding_base_url or "",
                model=self.settings.embedding_model or "",
            )
            self.llm_client = LLMClient(
                api_key=self.settings.llm_api_key or "",
                base_url=self.settings.llm_base_url or "",
                model=self.settings.llm_model or "",
            )
            self.vector_store = VectorStore(
                persist_dir=self.settings.vector_store_dir,
                collection_name=self.settings.collection_name,
            )
            self.mode = "rag"
            self._rebuild_vector_index()
        except Exception as exc:
            LOGGER.exception("Failed to initialize full RAG runtime.")
            self.startup_warning = f"完整 RAG 初始化失败，已切换到本地兜底模式：{exc}"
            self.mode = "local"

    def _rebuild_vector_index(self) -> None:
        if not self.embedding_generator or not self.vector_store:
            raise RuntimeError("RAG runtime is not initialized.")

        self.vector_store.reset()
        if not self.chunks:
            self.startup_warning = "知识库暂无文档。请把 .txt 或 .md 文件放入 data/documents 后重建索引。"
            return

        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedding_generator.generate_many(texts)
        self.vector_store.upsert_chunks(self.chunks, embeddings)
        if self.startup_warning and "知识库暂无文档" in self.startup_warning:
            self.startup_warning = None

    def _answer_with_rag(self, query: str) -> RAGResult:
        if not self.embedding_generator or not self.vector_store or not self.llm_client:
            raise RuntimeError("RAG runtime is not initialized.")

        query_embedding = self.embedding_generator.generate(query)
        sources = self.vector_store.query(query_embedding, self.settings.top_k)
        context = build_context(sources, self.settings.max_context_chars)

        system_prompt = (
            "你是一个严谨的中文 RAG 问答助手。只能基于给定的知识库上下文回答。"
            "如果上下文不足以回答，请明确说明不知道，并建议用户补充知识库文档。"
            "回答要简洁、准确，并尽量引用来源名称。"
        )
        user_prompt = (
            f"问题：{query}\n\n"
            f"知识库上下文：\n{context or '暂无可用上下文。'}"
        )
        answer = self.llm_client.generate(system_prompt, user_prompt)
        return RAGResult(answer=answer, sources=sources, mode="rag")
