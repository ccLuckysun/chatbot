from __future__ import annotations

import json
import math
from pathlib import Path

from rag.documents import DocumentChunk


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.persist_file = persist_dir / f"{collection_name}.json"
        self.entries = self._load_entries()

    def count(self) -> int:
        return len(self.entries)

    def reset(self) -> None:
        self.entries = []
        self._save_entries()

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        existing = {entry["id"]: entry for entry in self.entries}
        for chunk, embedding in zip(chunks, embeddings):
            existing[chunk.id] = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }

        self.entries = list(existing.values())
        self._save_entries()

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self.count() == 0:
            return []

        scored_entries = [
            (cosine_similarity(query_embedding, entry["embedding"]), entry)
            for entry in self.entries
        ]
        scored_entries.sort(key=lambda item: item[0], reverse=True)

        sources: list[dict] = []
        for score, entry in scored_entries[:top_k]:
            metadata = entry.get("metadata", {})
            sources.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "text": entry.get("text", ""),
                    "score": round(score, 3),
                }
            )

        return sources

    def _load_entries(self) -> list[dict]:
        if not self.persist_file.exists():
            return []

        try:
            data = json.loads(self.persist_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        return data if isinstance(data, list) else []

    def _save_entries(self) -> None:
        self.persist_file.write_text(
            json.dumps(self.entries, ensure_ascii=False),
            encoding="utf-8",
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
