from __future__ import annotations

from urllib.parse import urlparse

from langchain_openai import ChatOpenAI


def create_chat_model(api_key: str, base_url: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        base_url=normalize_openai_base_url(base_url),
        model=model,
        temperature=0.2,
    )


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
