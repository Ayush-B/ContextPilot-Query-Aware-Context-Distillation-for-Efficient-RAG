from __future__ import annotations

from contextpilot.graph.pipeline import run_budgeted_rag


def main() -> None:
    query = "What is hybrid retrieval?"
    result = run_budgeted_rag(query=query, k=5, budget=2, return_dict=True)

    print("\n=== BUDGETED RAG VALIDATION ===")
    print(f"Query: {result['query']}")
    print(f"Retrieved chunk count: {len(result['retrieved_chunks'])}")
    print(f"Selected chunk count: {len(result['selected_chunks'])}")
    print(f"Budget: {result['budget']}")

    prompt_preview = result["prompt"][:500].replace("\n", " ")
    print(f"\nPrompt preview:\n{prompt_preview}...")

    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()