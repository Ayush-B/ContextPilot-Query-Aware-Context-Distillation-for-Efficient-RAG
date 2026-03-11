from contextpilot.ingestion.loader import DocumentLoader


def main() -> None:
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        print("-" * 50)
        print(f"document_id: {doc.document_id}")
        print(f"title: {doc.title}")
        print(f"source: {doc.source}")
        print(f"preview: {doc.text[:120]}")


if __name__ == "__main__":
    main()