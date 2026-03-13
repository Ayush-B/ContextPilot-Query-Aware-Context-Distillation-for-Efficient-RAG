from contextpilot.retrieval.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    query = "What is hybrid retrieval?"
    results = retriever.retrieve(query=query, k=5)

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} results\n")

    for rank, result in enumerate(results, start=1):
        print("-" * 60)
        print(f"rank: {rank}")
        print(f"score: {result.score:.4f}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"document_id: {result.document_id}")
        print(f"chunk_index: {result.chunk_index}")
        print(f"title: {result.title}")
        print(f"source: {result.source}")
        print(f"text: {result.text[:250]}")
        print()


if __name__ == "__main__":
    main()