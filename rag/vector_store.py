from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag.documents import DocumentChunk


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self._get_or_create_collection()

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.collection = self._get_or_create_collection()
            return int(self.collection.count())

    def reset(self) -> None:
        try:
            self.collection = self._get_or_create_collection()
            existing = self.collection.get(include=[])
            ids = existing.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.collection = self._get_or_create_collection()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        try:
            self.collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=embeddings,
            )
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.collection = self._get_or_create_collection()
            self.collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=embeddings,
            )

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if not _is_missing_collection_error(exc):
                raise
            self.collection = self._get_or_create_collection()
            return []

        documents = _first(results.get("documents"))
        metadatas = _first(results.get("metadatas"))
        distances = _first(results.get("distances"))

        sources: list[dict] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            score = _distance_to_score(distance)
            sources.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index"),
                    "text": document or "",
                    "score": score,
                }
            )

        return sources

    def _get_or_create_collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _first(value: Any) -> list:
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, list) else []
    return []


def _distance_to_score(distance: float | int | None) -> float:
    if distance is None:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - float(distance))), 3)


def _is_missing_collection_error(exc: Exception) -> bool:
    message = str(exc)
    return "Collection [" in message and "does not exist" in message
