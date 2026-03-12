from contextpilot.ingestion.chunker import DocumentChunker
from contextpilot.ingestion.loader import DocumentLoader


def main() -> None:
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    chunker = DocumentChunker(chunk_size=80, chunk_overlap=20)
    chunks = chunker.chunk_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    for chunk in chunks[:10]:
        print("-" * 60)
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"document_id: {chunk.document_id}")
        print(f"chunk_index: {chunk.chunk_index}")
        print(f"title: {chunk.title}")
        print(f"source: {chunk.source}")
        print(f"characters: {len(chunk.text)}")
        print(f"preview: {chunk.text}")


if __name__ == "__main__":
    main()