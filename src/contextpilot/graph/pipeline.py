from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from contextpilot.generation.generator import Generator
from contextpilot.generation.prompt_builder import PromptBuilder
from contextpilot.retrieval.retriever import Retriever


@dataclass
class RawRAGResult:
    query: str
    retrieved_chunks: list[Any]
    prompt: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _serialize_chunk(chunk: Any) -> Any:
    """
    Best-effort serializer for retrieved chunk objects.

    This keeps the pipeline lightweight and avoids introducing duplicate
    schema logic. If the chunk is already a dict or primitive, it is returned
    as-is. If it is a dataclass or simple object, common fields are extracted.
    """
    if isinstance(chunk, (str, int, float, bool)) or chunk is None:
        return chunk

    if isinstance(chunk, dict):
        return chunk

    # dataclass-like objects
    if hasattr(chunk, "__dataclass_fields__"):
        return asdict(chunk)

    # common chunk attributes
    serialized: dict[str, Any] = {}
    for attr in ("text", "content", "chunk_id", "doc_id", "source", "metadata", "score"):
        if hasattr(chunk, attr):
            serialized[attr] = getattr(chunk, attr)

    if serialized:
        return serialized

    return str(chunk)


def run_raw_rag(query: str, k: int = 5, return_dict: bool = True) -> RawRAGResult | dict[str, Any]:
    """
    Orchestrates the Raw RAG pipeline by reusing existing modules only.

    Steps:
    1. retrieve top-k chunks
    2. build raw RAG prompt
    3. generate answer
    4. return structured result

    Args:
        query: User query.
        k: Number of chunks to retrieve.
        return_dict: If True, returns a JSON-friendly dict. Otherwise returns
            the RawRAGResult dataclass.

    Returns:
        RawRAGResult or dict with:
            - query
            - retrieved_chunks
            - prompt
            - answer
    """
    retriever = Retriever()
    prompt_builder = PromptBuilder()
    generator = Generator()

    retrieved_chunks = retriever.retrieve(query=query, k=k)
    prompt = prompt_builder.build_prompt(query=query, chunks=retrieved_chunks)
    answer = generator.generate(prompt)

    result = RawRAGResult(
        query=query,
        retrieved_chunks=retrieved_chunks,
        prompt=prompt,
        answer=answer,
    )

    if not return_dict:
        return result

    return {
        "query": result.query,
        "retrieved_chunks": [_serialize_chunk(chunk) for chunk in result.retrieved_chunks],
        "prompt": result.prompt,
        "answer": result.answer,
    }