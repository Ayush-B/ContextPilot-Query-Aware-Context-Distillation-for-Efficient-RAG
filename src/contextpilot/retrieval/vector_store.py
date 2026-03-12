from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSVectorStore:
    """
    FAISS-based dense vector store.

    Responsibilities:
    - embed chunk text
    - build FAISS index
    - persist index
    - persist metadata
    - search similar chunks
    """

    def __init__(
        self,
        index_dir: str | Path = "data/vector_store/faiss",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.pkl"

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def add_chunks(self, chunks: Iterable[Any]) -> None:
        """
        Embed and add chunks to the FAISS index.
        """

        chunk_list = list(chunks)
        if not chunk_list:
            return

        texts = [chunk.text for chunk in chunk_list]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        start_id = len(self.metadata)

        self.index.add(embeddings)

        for i, chunk in enumerate(chunk_list):
            self.metadata.append(
                {
                    "faiss_id": start_id + i,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                }
            )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Search the index using a text query.
        """

        if self.index is None:
            raise ValueError("FAISS index not initialized")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            item = self.metadata[idx].copy()
            item["score"] = float(score)

            results.append(item)

        return results

    def save(self) -> None:
        """
        Persist FAISS index and metadata.
        """

        if self.index is None:
            raise ValueError("Index is empty")

        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        """
        Load index and metadata from disk.
        """

        if not self.index_path.exists():
            raise FileNotFoundError("FAISS index not found")

        if not self.metadata_path.exists():
            raise FileNotFoundError("Metadata file not found")

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)