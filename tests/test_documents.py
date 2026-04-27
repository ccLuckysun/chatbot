from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag.documents import load_document_chunks, split_text, stable_chunk_id


class DocumentTests(unittest.TestCase):
    def test_stable_chunk_id_is_repeatable(self) -> None:
        first = stable_chunk_id("guide.md", 0, "hello")
        second = stable_chunk_id("guide.md", 0, "hello")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("guide.md:0:"))

    def test_split_text_uses_langchain_splitter(self) -> None:
        text = "第一段内容。" * 20
        chunks = split_text(text, chunk_size=30, chunk_overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))

    def test_load_document_chunks_returns_langchain_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text("第一段内容。\n\n第二段内容。", encoding="utf-8")

            documents = load_document_chunks(root, chunk_size=10, chunk_overlap=2)

        self.assertGreater(len(documents), 1)
        self.assertTrue(all(isinstance(document, Document) for document in documents))
        self.assertEqual(documents[0].metadata["source"], "guide.md")
        self.assertEqual(documents[0].metadata["chunk_index"], 0)
        self.assertTrue(documents[0].metadata["id"].startswith("guide.md:0:"))


if __name__ == "__main__":
    unittest.main()
