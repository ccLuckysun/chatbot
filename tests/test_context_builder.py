from __future__ import annotations

import unittest

from rag.context_builder import build_context


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_includes_sources_and_scores(self) -> None:
        context = build_context(
            [
                {
                    "source": "guide.md",
                    "score": 0.92,
                    "text": "RAG 包含检索和生成两个关键步骤。",
                }
            ],
            max_chars=500,
        )

        self.assertIn("source=guide.md", context)
        self.assertIn("score=0.92", context)
        self.assertIn("RAG 包含检索", context)

    def test_build_context_respects_max_chars(self) -> None:
        context = build_context(
            [{"source": "long.md", "text": "A" * 1000}],
            max_chars=220,
        )

        self.assertLessEqual(len(context), 220)
        self.assertIn("[truncated]", context)


if __name__ == "__main__":
    unittest.main()
