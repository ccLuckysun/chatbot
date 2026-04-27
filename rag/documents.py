from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".md", ".txt"}
TEXT_SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]


def load_document_chunks(
    directory: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    directory.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            continue

        relative_path = path.relative_to(directory).as_posix()
        source_document = Document(
            page_content=text,
            metadata={"source": relative_path},
        )

        for index, chunk in enumerate(text_splitter.split_documents([source_document])):
            chunk_text = chunk.page_content.strip()
            if not chunk_text:
                continue
            chunk_id = stable_chunk_id(relative_path, index, chunk_text)
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "id": chunk_id,
                    "source": relative_path,
                    "chunk_index": index,
                }
            )
            documents.append(Document(page_content=chunk_text, metadata=metadata))

    return documents


def create_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=TEXT_SEPARATORS,
        add_start_index=True,
    )


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    return [document.page_content for document in splitter.create_documents([normalized])]


def stable_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"{source}:{index}:{digest[:12]}"


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()
