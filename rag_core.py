"""Offline-testable lexical ranking and reciprocal-rank fusion."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    source: str
    page: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_rank(query: str, chunks: list[Chunk]) -> list[int]:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return []
    scores: list[tuple[float, int]] = []
    for chunk in chunks:
        terms = Counter(tokenize(chunk.text))
        length = max(sum(terms.values()), 1)
        score = sum(
            query_count * (1 + log(1 + terms.get(term, 0))) / length
            for term, query_count in query_terms.items()
            if term in terms
        )
        if score:
            scores.append((score, chunk.chunk_id))
    return [chunk_id for _, chunk_id in sorted(scores, reverse=True)]


def reciprocal_rank_fusion(
    dense_ids: list[int], lexical_ids: list[int], k: int = 60
) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in (dense_ids, lexical_ids):
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return [item[0] for item in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
