from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag.local_search import LocalSearchEngine
from rag.retriever import RAGRetriever


class FakeEmbeddingGenerator:
    def generate(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        return [
            {
                "source": "测试来源.md",
                "score": 0.88,
                "text": "这是用于 mock RAG 的上下文。",
            }
        ]


class FakeLLMClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "测试来源.md" not in user_prompt:
            raise AssertionError("user_prompt should include retrieved source context")
        return "这是 mock LLM 生成的回答。"


class RetrieverTests(unittest.TestCase):
    def test_answer_with_mock_rag_components(self) -> None:
        retriever = object.__new__(RAGRetriever)
        retriever.settings = SimpleNamespace(top_k=1, max_context_chars=500)
        retriever.startup_warning = None
        retriever.chunks = []
        retriever.local_engine = LocalSearchEngine([])
        retriever.embedding_generator = FakeEmbeddingGenerator()
        retriever.vector_store = FakeVectorStore()
        retriever.llm_client = FakeLLMClient()
        retriever.mode = "rag"

        result = retriever.answer("RAG 是什么？")

        self.assertEqual(result.mode, "rag")
        self.assertEqual(result.answer, "这是 mock LLM 生成的回答。")
        self.assertEqual(result.sources[0]["source"], "测试来源.md")


if __name__ == "__main__":
    unittest.main()
