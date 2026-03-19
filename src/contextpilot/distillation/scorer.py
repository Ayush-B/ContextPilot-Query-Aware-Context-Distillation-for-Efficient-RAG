from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "of", "in", "on", "for", "to", "from", "with", "by", "at", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "this", "that", "these", "those", "it", "its", "into", "about",
    "can", "could", "should", "would", "may", "might", "will", "shall",
    "do", "does", "did", "not", "only", "using", "used",
}


@dataclass
class ChunkScore:
    rank: int
    chunk: Any
    score: float
    overlap_count: int
    overlap_ratio: float
    query_terms: list[str]
    matched_terms: list[str]
    token_estimate: int
    provisional_label: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunk"] = _serialize_chunk(self.chunk)
        return data


def _serialize_chunk(chunk: Any) -> Any:
    if isinstance(chunk, (str, int, float, bool)) or chunk is None:
        return chunk

    if isinstance(chunk, dict):
        return chunk

    if hasattr(chunk, "__dataclass_fields__"):
        return asdict(chunk)

    serialized: dict[str, Any] = {}
    for attr in ("text", "content", "chunk_id", "doc_id", "source", "metadata", "score"):
        if hasattr(chunk, attr):
            serialized[attr] = getattr(chunk, attr)

    if serialized:
        return serialized

    return str(chunk)


def _extract_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        for key in ("text", "content"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value

    for attr in ("text", "content"):
        if hasattr(chunk, attr):
            value = getattr(chunk, attr)
            if isinstance(value, str):
                return value

    return str(chunk)


def _tokenize(text: str) -> list[str]:
    terms = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [term for term in terms if term not in STOPWORDS and len(term) > 1]


def _estimate_tokens(text: str) -> int:
    words = re.findall(r"\S+", text)
    return max(1, round(len(words) * 1.3))


def _assign_provisional_label(
    overlap_count: int,
    overlap_ratio: float,
    token_estimate: int,
) -> str:
    """
    Initial heuristic label for later distillation stages.

    quote:
        high relevance and concise enough to keep mostly as-is
    compress:
        useful, but likely too long or only moderately targeted
    drop:
        weak evidence for the current query
    """
    if overlap_count == 0:
        return "drop"

    if overlap_ratio >= 0.6 and token_estimate <= 120:
        return "quote"

    if overlap_ratio >= 0.3:
        return "compress"

    if overlap_count >= 2:
        return "compress"

    return "drop"


class EvidenceScorer:
    """
    Scores retrieved chunks against a query using deterministic heuristics.

    This is the first ContextPilot-specific module. It does not perform
    compression or evidence-pack assembly. It only produces structured
    scoring signals that later sprints can consume.
    """

    def score_chunks(self, query: str, chunks: list[Any]) -> list[ChunkScore]:
        query_terms = sorted(set(_tokenize(query)))
        query_term_set = set(query_terms)

        results: list[ChunkScore] = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_text = _extract_chunk_text(chunk)
            chunk_terms = set(_tokenize(chunk_text))

            matched_terms = sorted(query_term_set.intersection(chunk_terms))
            overlap_count = len(matched_terms)
            overlap_ratio = overlap_count / len(query_term_set) if query_term_set else 0.0
            token_estimate = _estimate_tokens(chunk_text)

            # Simple, interpretable score:
            # prioritize query coverage, then mildly reward conciseness
            brevity_bonus = 1.0 / max(token_estimate, 1)
            score = round((overlap_ratio * 0.9) + (overlap_count * 0.08) + (brevity_bonus * 2.0), 4)

            provisional_label = _assign_provisional_label(
                overlap_count=overlap_count,
                overlap_ratio=overlap_ratio,
                token_estimate=token_estimate,
            )

            results.append(
                ChunkScore(
                    rank=idx,
                    chunk=chunk,
                    score=score,
                    overlap_count=overlap_count,
                    overlap_ratio=round(overlap_ratio, 4),
                    query_terms=query_terms,
                    matched_terms=matched_terms,
                    token_estimate=token_estimate,
                    provisional_label=provisional_label,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)

        # Reassign ranks after sorting so output rank reflects score order
        reranked: list[ChunkScore] = []
        for new_rank, item in enumerate(results, start=1):
            reranked.append(
                ChunkScore(
                    rank=new_rank,
                    chunk=item.chunk,
                    score=item.score,
                    overlap_count=item.overlap_count,
                    overlap_ratio=item.overlap_ratio,
                    query_terms=item.query_terms,
                    matched_terms=item.matched_terms,
                    token_estimate=item.token_estimate,
                    provisional_label=item.provisional_label,
                )
            )

        return reranked