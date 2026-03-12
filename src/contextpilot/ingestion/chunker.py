from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken

from contextpilot.ingestion.loader import LoadedDocument


@dataclass(slots=True)
class ChunkedDocument:
    """
    A retrieval-ready chunk derived from a loaded document.

    Attributes:
        chunk_id: Stable unique identifier for the chunk.
        document_id: Identifier of the parent document.
        chunk_index: Position of the chunk within the parent document.
        title: Human-readable document title.
        source: Original source path.
        text: Chunk text.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    source: str
    text: str


class DocumentChunker:
    """
    Split loaded documents into overlapping token-based chunks.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_documents(
        self,
        documents: Iterable[LoadedDocument],
    ) -> list[ChunkedDocument]:
        """
        Chunk multiple loaded documents.

        Args:
            documents: Iterable of LoadedDocument instances.

        Returns:
            A flat list of ChunkedDocument objects.
        """
        all_chunks: list[ChunkedDocument] = []

        for document in documents:
            document_chunks = self.chunk_document(document)
            all_chunks.extend(document_chunks)

        return all_chunks

    def chunk_document(self, document: LoadedDocument) -> list[ChunkedDocument]:
        """
        Chunk a single loaded document into overlapping token windows.
        """
        token_ids = self.encoding.encode(document.text)
        if not token_ids:
            return []

        chunks: list[ChunkedDocument] = []
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start in range(0, len(token_ids), step):
            end = start + self.chunk_size
            chunk_token_ids = token_ids[start:end]

            if not chunk_token_ids:
                continue

            chunk_text = self.encoding.decode(chunk_token_ids).strip()
            if not chunk_text:
                continue

            chunks.append(
                ChunkedDocument(
                    chunk_id=f"{document.document_id}__chunk_{chunk_index}",
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    title=document.title,
                    source=document.source,
                    text=chunk_text,
                )
            )
            chunk_index += 1

            if end >= len(token_ids):
                break

        return chunks