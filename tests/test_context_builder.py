from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag.context_builder import build_context


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_includes_sources_and_scores(self) -> None:
        context = build_context(
            [
                (
                    Document(
                        page_content="RAG 包含检索和生成两个关键步骤。",
                        metadata={"source": "guide.md", "chunk_index": 0},
                    ),
                    0.92,
                )
            ],
            max_chars=500,
        )

        self.assertIn("source=guide.md", context)
        self.assertIn("score=0.92", context)
        self.assertIn("RAG 包含检索", context)

    def test_build_context_respects_max_chars(self) -> None:
        context = build_context(
            [
                (
                    Document(
                        page_content="A" * 1000,
                        metadata={"source": "long.md", "chunk_index": 0},
                    ),
                    0.8,
                )
            ],
            max_chars=220,
        )

        self.assertLessEqual(len(context), 220)
        self.assertIn("[truncated]", context)


if __name__ == "__main__":
    unittest.main()
