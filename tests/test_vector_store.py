from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.documents import DocumentChunk
from rag.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_upsert_query_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = VectorStore(Path(directory), "test_docs")
            try:
                chunks = [
                    DocumentChunk(
                        id="a",
                        text="alpha document",
                        metadata={"source": "a.md", "chunk_index": 0},
                    ),
                    DocumentChunk(
                        id="b",
                        text="beta document",
                        metadata={"source": "b.md", "chunk_index": 0},
                    ),
                ]

                store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
                results = store.query([1.0, 0.0], top_k=1)

                self.assertEqual(store.count(), 2)
                self.assertEqual(results[0]["source"], "a.md")
                self.assertEqual(results[0]["text"], "alpha document")
                self.assertGreaterEqual(results[0]["score"], 0.9)

                store.reset()
                self.assertEqual(store.count(), 0)
            finally:
                store.close()

    def test_count_recovers_when_collection_was_deleted_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = VectorStore(Path(directory), "test_docs")
            try:
                store.upsert_chunks(
                    [
                        DocumentChunk(
                            id="a",
                            text="alpha document",
                            metadata={"source": "a.md", "chunk_index": 0},
                        )
                    ],
                    [[1.0, 0.0]],
                )
                store.client.delete_collection("test_docs")

                self.assertEqual(store.count(), 0)

                store.upsert_chunks(
                    [
                        DocumentChunk(
                            id="b",
                            text="beta document",
                            metadata={"source": "b.md", "chunk_index": 0},
                        )
                    ],
                    [[0.0, 1.0]],
                )
                self.assertEqual(store.count(), 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
