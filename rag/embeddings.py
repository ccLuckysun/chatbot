from __future__ import annotations

from openai import OpenAI

from rag.llm import normalize_openai_base_url


class EmbeddingGenerator:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(base_url),
        )
        self.model = model

    def generate(self, text: str) -> list[float]:
        embeddings = self.generate_many([text])
        return embeddings[0]

    def generate_many(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([item.embedding for item in ordered])

        return embeddings
