from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vector_store import VectorStore


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        if "alpha" in normalized:
            return [1.0, 0.0]
        if "beta" in normalized:
            return [0.0, 1.0]
        return [0.5, 0.5]


class VectorStoreTests(unittest.TestCase):
    def test_upsert_query_and_reset(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store = VectorStore(Path(directory), "test_docs", TinyEmbeddings())
            try:
                documents = [
                    Document(
                        id="a",
                        page_content="alpha document",
                        metadata={"id": "a", "source": "a.md", "chunk_index": 0},
                    ),
                    Document(
                        id="b",
                        page_content="beta document",
                        metadata={"id": "b", "source": "b.md", "chunk_index": 0},
                    ),
                ]

                store.upsert_documents(documents)
                results = store.query("alpha query", top_k=1)

                self.assertEqual(store.count(), 2)
                self.assertEqual(results[0]["source"], "a.md")
                self.assertEqual(results[0]["text"], "alpha document")
                self.assertGreaterEqual(results[0]["score"], 0.9)

                store.reset()
                self.assertEqual(store.count(), 0)
            finally:
                store.close()

    def test_count_recovers_when_collection_was_deleted_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store = VectorStore(Path(directory), "test_docs", TinyEmbeddings())
            try:
                store.upsert_documents(
                    [
                        Document(
                            id="a",
                            page_content="alpha document",
                            metadata={"id": "a", "source": "a.md", "chunk_index": 0},
                        )
                    ]
                )
                store.client.delete_collection("test_docs")

                self.assertEqual(store.count(), 0)

                store.upsert_documents(
                    [
                        Document(
                            id="b",
                            page_content="beta document",
                            metadata={"id": "b", "source": "b.md", "chunk_index": 0},
                        )
                    ]
                )
                self.assertEqual(store.count(), 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
