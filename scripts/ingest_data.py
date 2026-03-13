import shutil
from pathlib import Path

from contextpilot.ingestion.loader import DocumentLoader
from contextpilot.ingestion.chunker import DocumentChunker
from contextpilot.retrieval.vector_store import FAISSVectorStore


def main() -> None:

    print("Starting ingestion pipeline\n")

    # Remove old index (optional but recommended)
    index_dir = Path("data/vector_store/faiss")

    if index_dir.exists():
        print("Removing old FAISS index...")
        shutil.rmtree(index_dir)

    # 1. Load documents
    print("\nLoading documents...")
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    print(f"Loaded {len(documents)} documents\n")

    # 2. Chunk documents
    print("Chunking documents...")
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_documents(documents)

    print(f"Created {len(chunks)} chunks\n")

    # 3. Build vector store
    print("Building FAISS index...")
    store = FAISSVectorStore()

    store.add_chunks(chunks)

    # 4. Save index
    store.save()

    print("\nIndex successfully created")
    print(f"Indexed {len(chunks)} chunks")
    print("Saved to data/vector_store/faiss/")


if __name__ == "__main__":
    main()