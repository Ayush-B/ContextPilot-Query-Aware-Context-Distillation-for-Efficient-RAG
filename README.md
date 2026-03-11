# ContextPilot

Query-Aware Context Distillation for Efficient Retrieval-Augmented Generation

## Overview

ContextPilot is an experimental Retrieval-Augmented Generation (RAG) system that reduces prompt size and retrieval overhead by distilling retrieved context before generation.

Traditional RAG pipelines retrieve several document chunks and pass them directly to a language model. This often increases token cost, latency, and the chance that irrelevant context distracts the model.

ContextPilot introduces an intermediate **context distillation stage** that evaluates retrieved chunks and decides whether they should be:

- quoted verbatim
- compressed into a shorter evidence statement
- dropped entirely

The remaining information is assembled into a compact **Evidence Pack**, which is used by the model to generate the final answer.

---

## Core Idea

Standard RAG pipeline:

Query → Retrieve Top-K Chunks → Send Chunks to LLM → Generate Answer

ContextPilot pipeline:

Query → Candidate Retrieval → Evidence Scoring → Distillation Decision (Quote | Compress | Drop) → Evidence Pack Assembly → LLM Generation

Instead of sending all retrieved chunks to the model, ContextPilot constructs a compact evidence set containing only the most relevant information.

---

## Distillation Strategy

Each retrieved chunk is evaluated using several attributes:

- **Relevance** – similarity to the query  
- **Novelty** – whether the chunk adds new information  
- **Technical Density** – amount of useful information relative to length  
- **Citation Value** – whether the chunk contains key definitions or explanations  

Based on these signals, chunks are classified into three actions:

**Quote**

High-relevance chunks that contain important technical details are kept unchanged.

**Compress**

Chunks with useful information but unnecessary verbosity are summarized.

**Drop**

Chunks that are redundant or weakly related to the query are removed.

---

## Example Evidence Pack

```
Evidence Pack

[Q1] Dense retrieval encodes queries and documents into vector space.
[C1] Hybrid retrieval combines keyword search with semantic embeddings.
[C2] Reranking improves final retrieval ordering.
```

The language model generates the final answer using this distilled evidence rather than the full retrieved context.

---

## Repository Structure

```
contextpilot/

src/
    config/
    ingestion/
    retrieval/
    distillation/
    generation/
    evaluation/
    graph/
    models/
    main.py

data/
notebooks/
tests/
```

---

## Development Roadmap

Sprint 0  
Project setup and architecture scaffold

Sprint 1  
Document ingestion and retrieval baseline

Sprint 2  
Baseline RAG pipelines

Sprint 3  
Evidence scoring

Sprint 4  
Context distillation

Sprint 5  
Grounded generation

Sprint 6  
Benchmark and evaluation

---

## Project Goal

The goal of ContextPilot is to explore whether **query-aware context distillation** can reduce prompt size and latency while preserving the grounding benefits of retrieval.

---
