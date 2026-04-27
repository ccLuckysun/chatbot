from __future__ import annotations

import math
import re
from collections import Counter

from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


class LocalSearchEngine:
    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self.chunk_terms = [Counter(tokenize(chunk.page_content)) for chunk in chunks]

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.chunks:
            return []

        query_terms = Counter(tokenize(query))
        if not query_terms:
            return self._format_results(
                [(0.0, index) for index in range(min(top_k, len(self.chunks)))]
            )

        scored: list[tuple[float, int]] = []
        for index, terms in enumerate(self.chunk_terms):
            score = self._score(query, query_terms, terms, self.chunks[index].page_content)
            if score > 0:
                scored.append((score, index))

        if not scored:
            scored = [(0.0, index) for index in range(min(top_k, len(self.chunks)))]

        scored.sort(key=lambda item: item[0], reverse=True)
        return self._format_results(scored[:top_k])

    def answer(self, query: str, top_k: int) -> tuple[str, list[dict]]:
        sources = self.search(query, top_k)
        if not sources:
            return (
                "知识库里还没有可检索的文档。请先把 .txt 或 .md 文件放入 data/documents。",
                [],
            )

        bullets = "\n".join(f"- {source['text'][:260].strip()}" for source in sources)
        answer = (
            "当前没有完整配置 LLM 与 Embedding API，已使用本地关键词检索兜底模式回答。\n\n"
            f"根据知识库中最相关的内容，可以参考：\n{bullets}\n\n"
            "如需完整 RAG 能力，请在 .env 中配置 LLM_* 与 EMBEDDING_* 后重启应用。"
        )
        return answer, sources

    @staticmethod
    def _score(query: str, query_terms: Counter, terms: Counter, text: str) -> float:
        overlap = 0.0
        total_terms = max(sum(terms.values()), 1)

        for token, query_count in query_terms.items():
            if token in terms:
                tf = terms[token] / total_terms
                overlap += min(query_count, terms[token]) * (1.0 + math.log1p(tf * 100))

        phrase_bonus = 3.0 if query.strip() and query.strip().lower() in text.lower() else 0.0
        return overlap + phrase_bonus

    def _format_results(self, scored: list[tuple[float, int]]) -> list[dict]:
        best_score = max((score for score, _ in scored), default=0.0) or 1.0
        results: list[dict] = []

        for score, index in scored:
            chunk = self.chunks[index]
            results.append(
                {
                    "source": chunk.metadata["source"],
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "text": chunk.page_content,
                    "score": round(score / best_score, 3),
                }
            )

        return results


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
