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
    api_key: str | None
    api_url: str | None
    model_name: str | None
    vector_store_dir: Path
    documents_dir: Path
    collection_name: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    log_file: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")

        return cls(
            api_key=_empty_to_none(os.getenv("API_KEY")),
            api_url=_empty_to_none(os.getenv("API_URL")),
            model_name=_empty_to_none(os.getenv("MODEL_NAME")),
            vector_store_dir=_resolve_path(
                os.getenv("VECTOR_STORE_DIR", os.getenv("CHROMA_PERSIST_DIR", "vector_store"))
            ),
            documents_dir=_resolve_path(os.getenv("DOCUMENTS_DIR", "data/documents")),
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", os.getenv("CHROMA_COLLECTION_NAME", "docs")),
            top_k=_get_int("TOP_K", 3),
            chunk_size=_get_int("CHUNK_SIZE", 900),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
            log_file=_resolve_path(os.getenv("LOG_FILE", "logs/app.log")),
        )

    @property
    def has_external_api_config(self) -> bool:
        return bool(self.api_key and self.api_url and self.model_name)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default
