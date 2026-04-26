from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict


def load_document_chunks(directory: Path, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    directory.mkdir(parents=True, exist_ok=True)
    chunks: list[DocumentChunk] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        relative_path = path.relative_to(directory).as_posix()
        for index, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap)):
            chunk_id = stable_chunk_id(relative_path, index, chunk_text)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": relative_path,
                        "chunk_index": index,
                    },
                )
            )

    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    overlap = max(0, min(chunk_overlap, chunk_size // 2))

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        soft_end = _find_soft_boundary(text, start, hard_end)
        chunk = text[start:soft_end].strip()
        if chunk:
            chunks.append(chunk)

        if soft_end >= len(text):
            break

        start = max(soft_end - overlap, start + 1)

    return chunks


def stable_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"{source}:{index}:{digest[:12]}"


def _find_soft_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)

    search_window = text[start:hard_end]
    boundary_candidates = [
        search_window.rfind("\n\n"),
        search_window.rfind("\n"),
        search_window.rfind("。"),
        search_window.rfind(". "),
    ]
    boundary = max(boundary_candidates)

    min_chunk = max(100, (hard_end - start) // 2)
    if boundary >= min_chunk:
        return start + boundary + 1

    return hard_end
