from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from rag.llm import normalize_openai_base_url


def create_embeddings(api_key: str, base_url: str, model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=normalize_openai_base_url(base_url),
        model=model,
        check_embedding_ctx_length=False,
    )
