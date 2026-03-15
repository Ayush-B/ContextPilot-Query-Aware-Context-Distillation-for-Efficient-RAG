# ContextPilot

Query-Aware Context Distillation for Efficient Retrieval-Augmented Generation

## Overview

ContextPilot is an experimental Retrieval-Augmented Generation (RAG) system designed to reduce prompt size, retrieval overhead, and latency by distilling retrieved context before generation.

Traditional RAG pipelines retrieve multiple document chunks and pass them directly to a language model. This often increases token cost, latency, and the chance that irrelevant context distracts the model.

ContextPilot introduces an intermediate context distillation stage that evaluates retrieved chunks and decides whether they should be:

- quoted verbatim
- compressed into a shorter evidence statement
- dropped entirely

The retained information is assembled into a compact Evidence Pack, which is then used to generate the final answer.

------------------------------------------------------------

## Research Question

Can a RAG system reduce prompt size and latency by distilling retrieved evidence while preserving grounded answer quality?

------------------------------------------------------------

## Core Idea

Standard RAG pipeline:

Query → Retrieve Top-K Chunks → Send Chunks to LLM → Generate Answer

ContextPilot pipeline:

Query → Candidate Retrieval → Evidence Scoring → Distillation Decision (Quote | Compress | Drop) → Evidence Pack Assembly → LLM Generation

Instead of sending all retrieved chunks to the model, ContextPilot constructs a compact evidence set containing only the most relevant information.

------------------------------------------------------------

## Current Implementation Status

Sprint 0 — Project Setup

Completed:
- repository initialization
- modular src package structure
- configuration system
- pipeline schemas
- project entrypoint

Sprint 1 — Retrieval Foundation

Completed:
- document loader for .txt and .md files
- token-based chunking pipeline
- local dense retrieval using BAAI/bge-small-en-v1.5
- FAISS vector store
- retriever interface
- ingestion pipeline script
- validation scripts for loader, chunker, and retriever

Retrieval pipeline:

Documents
→ Loader
→ Chunker
→ Embeddings
→ FAISS Index
→ Retriever

Sprint 2 — Baseline RAG Pipelines (In Progress)

Implemented:
- Raw RAG prompt builder
- LLM generation wrapper
- provider abstraction for OpenAI or Gemini
- end-to-end Raw RAG validation

Baseline pipeline:

Query
→ Retriever
→ Prompt Builder
→ LLM Generator
→ Answer

------------------------------------------------------------

## Retrieval Stack

Current retrieval architecture includes:

- document ingestion from data/raw
- token-based chunking
- embeddings using BAAI/bge-small-en-v1.5
- FAISS similarity search
- top-k semantic retrieval

Key technologies:

- FAISS for vector indexing
- SentenceTransformers embeddings
- modular retriever interface
- structured chunk metadata

------------------------------------------------------------

## Planned Distillation Strategy

ContextPilot will evaluate each retrieved chunk using several signals:

Relevance
Similarity to the query.

Novelty
Whether the chunk introduces new information compared to previously selected chunks.

Technical Density
Amount of useful information relative to chunk length.

Citation Value
Whether the chunk contains important definitions, explanations, or references.

Based on these signals, chunks are assigned one of three actions.

Quote
High-value chunks containing important technical details are kept unchanged.

Compress
Chunks containing useful information but unnecessary verbosity are summarized into shorter evidence statements.

Drop
Chunks that are redundant or weakly related to the query are removed.

------------------------------------------------------------

## Example Evidence Pack

Evidence Pack

[Q1] Dense retrieval encodes queries and documents into vector space.
[C1] Hybrid retrieval combines keyword search with semantic embeddings.
[C2] Reranking improves final retrieval ordering.

The final answer will be generated from this distilled evidence set instead of the full retrieved context.

------------------------------------------------------------

## Repository Structure

.
├── data
│   ├── raw
│   ├── processed
│   ├── eval
│   └── vector_store
│
├── notebooks
├── scripts
├── tests
│
├── src
│   └── contextpilot
│       ├── config
│       ├── ingestion
│       ├── retrieval
│       ├── generation
│       ├── distillation
│       ├── evaluation
│       ├── graph
│       ├── models
│       └── main.py
│
├── pyproject.toml
├── README.md
└── .env.example

------------------------------------------------------------

## Quickstart

1. Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\Activate.ps1

2. Install the project

pip install -e .

3. Configure environment variables

Copy .env.example to .env

copy .env.example .env

Edit .env and add your API keys:

OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

4. Build the FAISS index

python scripts/ingest_data.py

5. Validate retrieval

python scripts/validate_retriever.py

6. Validate Raw RAG baseline

python scripts/validate_raw_rag.py

------------------------------------------------------------

## Development Roadmap

Sprint 0
Project setup and architecture scaffold

Sprint 1
Document ingestion and dense retrieval system

Sprint 2
Baseline RAG pipelines

Sprint 3
Evidence scoring

Sprint 4
Context distillation

Sprint 5
Grounded generation

Sprint 6
Benchmarking and evaluation

------------------------------------------------------------

## Project Goal

The goal of ContextPilot is to investigate whether query-aware context distillation can reduce prompt size and latency while preserving the grounding benefits of retrieval.

The project aims to explore a retrieval architecture that is both efficient and grounded, enabling language models to operate with smaller, more informative context windows.