from __future__ import annotations


def build_context(sources: list[dict], max_chars: int) -> str:
    if not sources:
        return ""

    pieces: list[str] = []
    used_chars = 0
    separator = "\n\n"

    for index, source in enumerate(sources, start=1):
        source_name = str(source.get("source") or "unknown")
        score = source.get("score")
        score_text = f", score={score}" if isinstance(score, (int, float)) else ""
        header = f"[{index}] source={source_name}{score_text}\n"
        text = str(source.get("text") or "").strip()
        piece = f"{header}{text}"

        extra_separator = len(separator) if pieces else 0
        remaining = max_chars - used_chars - extra_separator
        if max_chars > 0 and remaining <= 0:
            break

        if max_chars > 0 and len(piece) > remaining:
            available_text_chars = remaining - len(header) - 14
            if available_text_chars < 120:
                break
            piece = f"{header}{text[:available_text_chars].rstrip()}\n[truncated]"

        if pieces:
            used_chars += len(separator)
        pieces.append(piece)
        used_chars += len(piece)

    return separator.join(pieces)
