from __future__ import annotations

from contextpilot.distillation.scorer import EvidenceScorer
from contextpilot.retrieval.retriever import Retriever


def _preview_chunk(chunk: object, limit: int = 140) -> str:
    if isinstance(chunk, str):
        text = chunk
    elif isinstance(chunk, dict):
        text = str(chunk.get("text") or chunk.get("content") or chunk)
    else:
        text = str(getattr(chunk, "text", None) or getattr(chunk, "content", None) or chunk)

    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def main() -> None:
    query = "What is hybrid retrieval?"

    retriever = Retriever()
    chunks = retriever.retrieve(query=query, k=5)

    scorer = EvidenceScorer()
    scored_chunks = scorer.score_chunks(query=query, chunks=chunks)

    print("\n=== EVIDENCE SCORER VALIDATION ===")
    print(f"Query: {query}")
    print(f"Retrieved chunk count: {len(chunks)}")
    print(f"Scored chunk count: {len(scored_chunks)}")

    for item in scored_chunks:
        print("\n------------------------------------------------------------")
        print(f"Rank: {item.rank}")
        print(f"Score: {item.score}")
        print(f"Overlap count: {item.overlap_count}")
        print(f"Overlap ratio: {item.overlap_ratio}")
        print(f"Matched terms: {item.matched_terms}")
        print(f"Token estimate: {item.token_estimate}")
        print(f"Provisional label: {item.provisional_label}")
        print(f"Chunk preview: {_preview_chunk(item.chunk)}")


if __name__ == "__main__":
    main()