from __future__ import annotations

from contextpilot.graph.pipeline import run_raw_rag


def main() -> None:
    query = "What is hybrid retrieval?"
    result = run_raw_rag(query=query, k=5, return_dict=True)

    print("\n=== PIPELINE VALIDATION ===")
    print(f"Query: {result['query']}")
    print(f"Retrieved chunk count: {len(result['retrieved_chunks'])}")

    prompt_preview = result["prompt"][:500].replace("\n", " ")
    print(f"\nPrompt preview:\n{prompt_preview}...")

    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()