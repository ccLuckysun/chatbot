from __future__ import annotations

from urllib.parse import urlparse

from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(base_url),
        )
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        message = response.choices[0].message
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise RuntimeError("LLM response did not contain generated text.")


def normalize_openai_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Base URL must be a valid http(s) URL.")

    for suffix in ("/chat/completions", "/embeddings", "/responses"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("/")

    if not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"

    return cleaned
