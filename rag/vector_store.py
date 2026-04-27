from __future__ import annotations

import gc
from pathlib import Path

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_function: Embeddings,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.store = self._create_store()
        self.client = self.store._client

    def count(self) -> int:
        try:
            return int(self.store._collection.count())
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.store = self._create_store()
            self.client = self.store._client
            return int(self.store._collection.count())

    def reset(self) -> None:
        try:
            existing = self.store.get(include=[])
            ids = existing.get("ids", [])
            if ids:
                self.store.delete(ids=ids)
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.store = self._create_store()
            self.client = self.store._client

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        self.store = None
        self.client = None
        SharedSystemClient.clear_system_cache()
        gc.collect()

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        ids = [str(document.metadata.get("id") or "") for document in documents]
        if not all(ids):
            raise ValueError("Every document must include metadata['id'].")

        try:
            self.store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.store = self._create_store()
            self.client = self.store._client
            self.store.add_documents(documents=documents, ids=ids)

    def query(self, query: str, top_k: int) -> list[dict]:
        if self.count() == 0:
            return []

        try:
            results = self.store.similarity_search_with_score(query, k=max(1, top_k))
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.store = self._create_store()
            self.client = self.store._client
            return []

        return documents_with_scores_to_sources(results)

    def _create_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )


def documents_with_scores_to_sources(results: list[tuple[Document, float]]) -> list[dict]:
    sources: list[dict] = []
    for document, distance in results:
        metadata = document.metadata or {}
        sources.append(
            {
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index"),
                "text": document.page_content or "",
                "score": _distance_to_score(distance),
            }
        )
    return sources


def _distance_to_score(distance: float | int | None) -> float:
    if distance is None:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - float(distance))), 3)


def _is_missing_collection_error(exc: Exception) -> bool:
    message = str(exc)
    return "Collection [" in message and "does not exist" in message
