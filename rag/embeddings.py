from __future__ import annotations

from openai import OpenAI


class EmbeddingGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, text: str) -> list[float]:
        embeddings = self.generate_many([text])
        return embeddings[0]

    def generate_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
