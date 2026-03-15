from __future__ import annotations

from typing import List

from contextpilot.retrieval.retriever import RetrievalResult


class PromptBuilder:
    """
    Builds prompts for the Raw RAG baseline.

    The prompt consists of:
    - system instruction
    - retrieved context chunks
    - user question
    """

    def __init__(self, max_chunks: int = 5) -> None:
        self.max_chunks = max_chunks

    def build_prompt(self, query: str, chunks: List[RetrievalResult]) -> str:
        """
        Construct a prompt using retrieved chunks.

        Args:
            query: user question
            chunks: retrieved context chunks

        Returns:
            formatted prompt string
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        selected_chunks = chunks[: self.max_chunks]

        context_sections = []

        for i, chunk in enumerate(selected_chunks, start=1):

            context_sections.append(
                f"[{i}] Source: {chunk.document_id}\n{chunk.text}"
            )

        context_block = "\n\n".join(context_sections)

        prompt = f"""
You are a helpful assistant. Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say that the information is not available.

Question:
{query}

Context:
{context_block}

Answer:
""".strip()

        return prompt