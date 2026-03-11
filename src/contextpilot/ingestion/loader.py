from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(slots=True)
class LoadedDocument:
    """
    Normalized document representation produced by the ingestion loader.

    Attributes:
        document_id: Stable identifier derived from the relative file path.
        title: Human-readable document title, usually from the filename stem.
        text: Full raw document text.
        source: File path string pointing to the original source document.
    """

    document_id: str
    title: str
    text: str
    source: str


class DocumentLoader:
    """
    Load raw text and markdown documents from a directory tree.

    This class is intentionally narrow in scope:
    - discovers supported files
    - reads file contents
    - normalizes whitespace lightly
    - returns structured LoadedDocument objects

    Chunking, embedding, and indexing are handled elsewhere.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()

    def load_documents(self) -> list[LoadedDocument]:
        """
        Recursively load all supported documents from the data directory.

        Returns:
            A list of LoadedDocument objects.

        Raises:
            FileNotFoundError: If the data directory does not exist.
            NotADirectoryError: If the provided path is not a directory.
        """
        self._validate_data_dir()

        documents: list[LoadedDocument] = []
        for file_path in self._iter_supported_files():
            text = self._read_text_file(file_path)
            if not text.strip():
                continue

            relative_path = file_path.relative_to(self.data_dir)
            documents.append(
                LoadedDocument(
                    document_id=self._build_document_id(relative_path),
                    title=file_path.stem,
                    text=self._normalize_text(text),
                    source=str(file_path),
                )
            )

        return documents

    def _validate_data_dir(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.data_dir}")

    def _iter_supported_files(self) -> Iterable[Path]:
        files = sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return files

    @staticmethod
    def _read_text_file(file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Apply light normalization without destroying structure.

        Current choices:
        - normalize Windows newlines
        - strip leading/trailing whitespace
        - preserve paragraph breaks
        """
        return text.replace("\r\n", "\n").strip()

    @staticmethod
    def _build_document_id(relative_path: Path) -> str:
        """
        Create a stable document id from the file's relative path.

        Example:
            sample_docs/rag_intro.txt -> sample_docs__rag_intro
        """
        parts = list(relative_path.with_suffix("").parts)
        return "__".join(parts)