from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from contextpilot.retrieval.vector_store import FAISSVectorStore


@dataclass(slots=True)
class RetrievalResult:
    """
    A structured retrieval result returned by the retriever interface.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    source: str
    text: str
    score: float


class Retriever:
    """
    High-level retrieval interface for ContextPilot.

    Responsibilities:
    - load a persisted FAISS vector store
    - execute semantic search
    - return structured retrieval results
    """

    def __init__(
        self,
        index_dir: str | Path = "data/vector_store/faiss",
        model_name: str = "BAAI/bge-small-en-v1.5",
        auto_load: bool = True,
    ) -> None:
        self.vector_store = FAISSVectorStore(
            index_dir=index_dir,
            model_name=model_name,
        )

        if auto_load:
            self.vector_store.load()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """
        Retrieve top-k semantically relevant chunks for a query.

        Args:
            query: User query string.
            k: Number of results to return.

        Returns:
            A list of RetrievalResult objects ordered by similarity.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        raw_results = self.vector_store.search(query=query, k=k)

        return [
            RetrievalResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                chunk_index=result["chunk_index"],
                title=result["title"],
                source=result["source"],
                text=result["text"],
                score=result["score"],
            )
            for result in raw_results
        ]