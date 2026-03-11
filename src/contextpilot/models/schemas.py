from typing import Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    source_path: str
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    retrieval_score: float


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    retrieval_score: float
    relevance: float
    novelty: float
    technical_density: float
    citation_value: float
    keep_mode: Literal["quote", "compress", "drop"]


class EvidenceItem(BaseModel):
    evidence_id: str
    mode: Literal["quote", "compress"]
    text: str
    source_chunk_ids: list[str]
    scores: dict = Field(default_factory=dict)


class EvidencePack(BaseModel):
    query: str
    items: list[EvidenceItem] = Field(default_factory=list)
    total_source_chunks: int = 0
    total_evidence_items: int = 0


class GeneratedAnswer(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    grounded: bool = False


class BenchmarkResult(BaseModel):
    system_name: str
    query: str
    prompt_tokens: int
    latency_ms: float
    grounded_support_score: float
    answer: str