from __future__ import annotations

import unittest

from rag.documents import split_text, stable_chunk_id


class DocumentTests(unittest.TestCase):
    def test_stable_chunk_id_is_repeatable(self) -> None:
        first = stable_chunk_id("guide.md", 0, "hello")
        second = stable_chunk_id("guide.md", 0, "hello")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("guide.md:0:"))

    def test_split_text_uses_overlap(self) -> None:
        text = "第一段内容。" * 20
        chunks = split_text(text, chunk_size=30, chunk_overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
