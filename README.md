# ContextPilot

Query-Aware Context Distillation for Efficient Retrieval-Augmented Generation

## Overview

ContextPilot is an experimental Retrieval-Augmented Generation (RAG) system designed to reduce prompt size and retrieval overhead while preserving grounded answer quality.

Traditional RAG pipelines often retrieve several chunks and pass them directly to a language model. This increases token cost, latency, and the chance that irrelevant context distracts generation.

ContextPilot introduces an intermediate **context distillation** stage between retrieval and generation. Instead of sending raw retrieved chunks directly to the model, it builds a compact **evidence pack** by deciding whether each chunk should be:

- quoted verbatim
- compressed into a shorter evidence statement
- dropped entirely

The final answer is generated from the distilled evidence pack rather than the full raw retrieval output.

## Research Question

Can a RAG system reduce prompt size and latency by distilling retrieved evidence while preserving grounded answer quality?

## Core Idea

Standard RAG pipeline:

Query
↓
Retrieve top-k chunks
↓
Send all chunks to LLM
↓
Generate answer
Query
↓
Candidate Retrieval
↓
Evidence Scoring
↓
Distillation Decision
    ├─ Quote
    ├─ Compress
    └─ Drop
↓
Evidence Pack Assembly
↓
LLM Generation
