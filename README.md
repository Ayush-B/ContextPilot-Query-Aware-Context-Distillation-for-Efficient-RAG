# ContextPilot
Query-Aware Context Distillation for Efficient Retrieval-Augmented Generation (RAG)

ContextPilot is an experimental Retrieval-Augmented Generation system that explores whether query-aware context distillation can reduce prompt size and latency while preserving grounded answer quality.

---

## Key Features

• Modular RAG architecture  
• Local dense retrieval using BGE embeddings and FAISS  
• Provider-agnostic generation (OpenAI or Gemini)  
• Experiment-ready pipeline for retrieval research  
• Planned context distillation stage for prompt efficiency  

---

## Overview

Traditional RAG pipelines retrieve several document chunks and send them directly to a language model. This increases prompt size, latency, and the chance that irrelevant context distracts the model.

ContextPilot introduces a context distillation stage that evaluates retrieved chunks and decides whether they should be:

• quoted verbatim  
• compressed into shorter evidence statements  
• dropped entirely  

The remaining information is assembled into a compact Evidence Pack, which is then used by the language model to generate the final answer.

---

## Research Question

Can a RAG system reduce prompt size and latency by distilling retrieved evidence while preserving grounded answer quality?

---

## System Architecture

Current Raw RAG pipeline:

```
User Query
    -> Retriever (FAISS + BGE embeddings)
    -> Retrieved Chunks
    -> Prompt Builder
    -> LLM Generator (OpenAI / Gemini)
    -> Generated Answer
```

Planned ContextPilot pipeline:

```
Query
    -> Candidate Retrieval
    -> Evidence Scoring
    -> Distillation Decision (Quote | Compress | Drop)
    -> Evidence Pack
    -> LLM Generation
    -> Final Answer
```

---

## Current Implementation Status

### Sprint 0 — Project Setup

Completed

• repository initialization  
• modular src package layout  
• configuration system  
• project schemas and settings  
• working project entrypoint  

---

### Sprint 1 — Retrieval Foundation

Completed

• document loader for .txt and .md files  
• token-based chunking pipeline  
• embeddings using BAAI/bge-small-en-v1.5  
• FAISS vector store with metadata  
• semantic retriever interface  
• ingestion pipeline script  
• validation scripts for loader, chunker, and retriever  

Retrieval pipeline

```
Documents
    -> Loader
    -> Chunker
    -> Embeddings
    -> FAISS Index
    -> Retriever
```

---

### Sprint 2 — Baseline RAG Pipelines

Implemented

• Raw RAG prompt builder  
• LLM generation wrapper  
• provider abstraction for OpenAI and Gemini  
• end-to-end Raw RAG validation  

Baseline pipeline

```
Query
    -> Retriever
    -> Prompt Builder
    -> LLM Generator
    -> Answer
```

---

## Retrieval Stack

The retrieval system uses a fully local dense retrieval architecture.

Components

• Document ingestion from data/raw  
• Token-based chunking  
• Dense embeddings using BAAI/bge-small-en-v1.5  
• FAISS similarity search  
• Top-k semantic retrieval  

Technologies

• FAISS for vector indexing  
• SentenceTransformers embeddings  
• modular retriever interface  
• structured metadata for traceable sources  

---

## Example Run

Example query

```
What is hybrid retrieval?
```

Model output

```
Hybrid retrieval combines dense retrieval and sparse retrieval. Dense retrieval uses vector embeddings to capture semantic similarity, while sparse retrieval methods such as BM25 rely on keyword overlap. Combining both approaches improves recall because the system can capture both semantic meaning and exact term matches.
```

This answer is generated from retrieved context rather than model parameters alone.

---

## Planned Distillation Strategy

ContextPilot will evaluate each retrieved chunk using several signals.

Relevance  
Similarity between query and chunk.

Novelty  
Whether the chunk introduces new information.

Technical Density  
Amount of useful information relative to chunk length.

Citation Value  
Whether the chunk contains key explanations or definitions.

Chunks are then classified into three actions.

Quote  
High-value chunks containing important technical details are kept unchanged.

Compress  
Chunks with useful information but unnecessary verbosity are summarized.

Drop  
Chunks that are redundant or weakly related to the query are removed.

---

## Example Evidence Pack

```
Evidence Pack

[Q1] Dense retrieval encodes queries and documents into vector space.
[C1] Hybrid retrieval combines keyword search with semantic embeddings.
[C2] Reranking improves final retrieval ordering.
```

The final answer will be generated from this distilled evidence instead of the full retrieved context.

---

## Repository Structure

```
.
├─ data
│  ├─ raw
│  ├─ processed
│  ├─ eval
│  └─ vector_store
│
├─ notebooks
├─ scripts
├─ tests
│
├─ src
│  └─ contextpilot
│     ├─ config
│     ├─ ingestion
│     ├─ retrieval
│     ├─ generation
│     ├─ distillation
│     ├─ evaluation
│     ├─ graph
│     ├─ models
│     └─ main.py
│
├─ pyproject.toml
├─ README.md
└─ .env.example
```

---

## Quickstart

Create and activate a virtual environment

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project

```
pip install -e .
```

Configure environment variables

```
copy .env.example .env
```

Add API keys inside `.env`.

Build the FAISS index

```
python scripts/ingest_data.py
```

Validate retrieval

```
python scripts/validate_retriever.py
```

Run the Raw RAG baseline

```
python scripts/validate_raw_rag.py
```

---

## Development Roadmap

Sprint 0  
Project setup and architecture scaffold

Sprint 1  
Document ingestion and dense retrieval

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

---

## Project Goal

ContextPilot investigates whether query-aware context distillation can reduce prompt size and latency while preserving the grounding benefits of retrieval.

The long-term objective is to design retrieval pipelines that deliver smaller, more informative context windows for language models while maintaining factual reliability.