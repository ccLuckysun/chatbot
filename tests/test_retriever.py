from __future__ import annotations

import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

from rag.local_search import LocalSearchEngine
from rag.retriever import RAGRetriever


class FakeVectorStore:
    def query(self, query: str, top_k: int) -> list[dict]:
        return [
            {
                "source": "测试来源.md",
                "score": 0.88,
                "text": "这是用于 mock RAG 的上下文。",
            }
        ]


class FakeRAGChain:
    def invoke(self, payload: dict) -> str:
        if "测试来源.md" not in payload["context"]:
            raise AssertionError("context should include retrieved source context")
        return "这是 mock LangChain RAG 生成的回答。"


class RetrieverTests(unittest.TestCase):
    def test_answer_with_mock_langchain_rag_components(self) -> None:
        retriever = object.__new__(RAGRetriever)
        retriever.settings = SimpleNamespace(top_k=1, max_context_chars=500)
        retriever.startup_warning = None
        retriever.chunks = []
        retriever.local_engine = LocalSearchEngine([])
        retriever.embedding_model = None
        retriever.vector_store = FakeVectorStore()
        retriever.llm_client = None
        retriever.rag_chain = FakeRAGChain()
        retriever.mode = "rag"

        result = retriever.answer("RAG 是什么？")

        self.assertEqual(result.mode, "rag")
        self.assertEqual(result.answer, "这是 mock LangChain RAG 生成的回答。")
        self.assertEqual(result.sources[0]["source"], "测试来源.md")

    def test_local_fallback_uses_langchain_documents(self) -> None:
        engine = LocalSearchEngine(
            [
                Document(
                    page_content="知识库可以通过添加 Markdown 文件扩展。",
                    metadata={"source": "guide.md", "chunk_index": 0},
                )
            ]
        )

        answer, sources = engine.answer("知识库", top_k=1)

        self.assertIn("本地关键词检索兜底模式", answer)
        self.assertEqual(sources[0]["source"], "guide.md")


if __name__ == "__main__":
    unittest.main()
