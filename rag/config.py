from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str | None
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str | None
    vector_store_dir: Path
    documents_dir: Path
    collection_name: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_context_chars: int
    log_file: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")

        return cls(
            llm_api_key=_env_value("LLM_API_KEY"),
            llm_base_url=_env_value("LLM_BASE_URL"),
            llm_model=_env_value("LLM_MODEL"),
            embedding_api_key=_env_value("EMBEDDING_API_KEY"),
            embedding_base_url=_env_value("EMBEDDING_BASE_URL"),
            embedding_model=_env_value("EMBEDDING_MODEL"),
            vector_store_dir=_resolve_path(
                os.getenv("VECTOR_STORE_DIR", os.getenv("CHROMA_PERSIST_DIR", "vector_store"))
            ),
            documents_dir=_resolve_path(os.getenv("DOCUMENTS_DIR", "data/documents")),
            collection_name=os.getenv(
                "VECTOR_COLLECTION_NAME",
                os.getenv("CHROMA_COLLECTION_NAME", "docs"),
            ),
            top_k=_get_int("TOP_K", 3),
            chunk_size=_get_int("CHUNK_SIZE", 900),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
            max_context_chars=_get_int("MAX_CONTEXT_CHARS", 6000),
            log_file=_resolve_path(os.getenv("LOG_FILE", "logs/app.log")),
        )

    @property
    def has_llm_config(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def has_embedding_config(self) -> bool:
        return bool(
            self.embedding_api_key
            and self.embedding_base_url
            and self.embedding_model
        )

    @property
    def has_rag_config(self) -> bool:
        return self.has_llm_config and self.has_embedding_config

    @property
    def missing_rag_config(self) -> list[str]:
        required = {
            "LLM_API_KEY": self.llm_api_key,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL": self.llm_model,
            "EMBEDDING_API_KEY": self.embedding_api_key,
            "EMBEDDING_BASE_URL": self.embedding_base_url,
            "EMBEDDING_MODEL": self.embedding_model,
        }
        return [name for name, value in required.items() if not value]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _env_value(name: str) -> str | None:
    value = _empty_to_none(os.getenv(name))
    if value is None or _is_template_value(value):
        return None
    return value


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _is_template_value(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized.startswith("your_")
        or normalized.startswith("your-")
        or normalized in {"https://api.example.com", "https://api.example.com/v1"}
        or "填入" in value
        or "填写" in value
    )


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default
