from contextpilot.ingestion.loader import DocumentLoader
from contextpilot.ingestion.chunker import DocumentChunker
from contextpilot.retrieval.vector_store import FAISSVectorStore


def main():

    loader = DocumentLoader("data/raw")
    docs = loader.load_documents()

    chunker = DocumentChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_documents(docs)

    store = FAISSVectorStore()

    store.add_chunks(chunks)
    store.save()

    print(f"Indexed {len(chunks)} chunks")

    results = store.search("What is FAISS used for?", k=5)

    print("\nTop results\n")

    for r in results:
        print("-" * 60)
        print("score:", r["score"])
        print("chunk_id:", r["chunk_id"])
        print("source:", r["source"])
        print("text:", r["text"][:200])


if __name__ == "__main__":
    main()