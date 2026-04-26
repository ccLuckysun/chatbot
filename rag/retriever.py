from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.config import Settings
from rag.documents import DocumentChunk, load_document_chunks
from rag.external_api import ExternalAPIClient
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
        self.external_client: ExternalAPIClient | None = None
        self.vector_store: VectorStore | None = None
        self.mode = "local"

        self._load_documents()
        self._configure_runtime()

    @property
    def mode_label(self) -> str:
        if self.mode == "external":
            return "外部 API RAG"
        return "本地兜底"

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def answer(self, query: str) -> RAGResult:
        if self.mode == "external":
            try:
                return self._answer_with_external_api(query)
            except Exception as exc:
                LOGGER.exception("External API RAG failed; falling back to local mode.")
                answer, sources = self.local_engine.answer(query, self.settings.top_k)
                answer = f"外部 API 调用失败，已临时切换到本地兜底模式。\n\n错误摘要：{exc}\n\n{answer}"
                return RAGResult(answer=answer, sources=sources, mode="local")

        answer, sources = self.local_engine.answer(query, self.settings.top_k)
        return RAGResult(answer=answer, sources=sources, mode="local")

    def rebuild_index(self) -> None:
        self._load_documents()

    def _load_documents(self) -> None:
        self.chunks = load_document_chunks(
            self.settings.documents_dir,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        self.local_engine = LocalSearchEngine(self.chunks)

    def _configure_runtime(self) -> None:
        if not self.settings.has_external_api_config:
            self.startup_warning = "未完整配置 API_KEY、API_URL、MODEL_NAME，应用正在使用本地兜底模式。"
            return

        try:
            self.external_client = ExternalAPIClient(
                api_key=self.settings.api_key or "",
                api_url=self.settings.api_url or "",
                model_name=self.settings.model_name or "",
            )
            self.vector_store = VectorStore(
                persist_dir=self.settings.vector_store_dir,
                collection_name=self.settings.collection_name,
            )
            self.mode = "external"
        except Exception as exc:
            LOGGER.exception("Failed to initialize external API runtime.")
            self.startup_warning = f"外部 API 初始化失败，已切换本地兜底模式：{exc}"
            self.mode = "local"

    def _answer_with_external_api(self, query: str) -> RAGResult:
        if not self.external_client:
            raise RuntimeError("External API runtime is not initialized.")

        sources = self.local_engine.search(query, self.settings.top_k)
        context = "\n\n".join(
            f"[来源: {source['source']}]\n{source['text']}" for source in sources
        )

        system_prompt = (
            "你是一个严谨的中文 RAG 助手。请只基于提供的知识库上下文回答。"
            "如果上下文不足，请明确说明不知道，并建议用户补充文档。回答要简洁、实用，并尽量引用来源名称。"
        )
        user_prompt = f"问题：{query}\n\n知识库上下文：\n{context or '暂无可用上下文。'}"
        answer = self.external_client.generate(system_prompt, user_prompt)
        return RAGResult(answer=answer, sources=sources, mode="external")
