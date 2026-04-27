from __future__ import annotations

import unittest

from rag.embeddings import create_embeddings


class EmbeddingFactoryTests(unittest.TestCase):
    def test_openai_compatible_embeddings_keep_raw_string_inputs(self) -> None:
        embeddings = create_embeddings(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="test-embedding-model",
        )

        self.assertFalse(embeddings.check_embedding_ctx_length)
        self.assertEqual(embeddings.model, "test-embedding-model")


if __name__ == "__main__":
    unittest.main()
