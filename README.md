# Precision Agriculture Water Management RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Ollama](https://img.shields.io/badge/Ollama-LLM-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An advanced Graph-Based Agentic RAG (GA-RAG) system for precision agriculture and water management, featuring semantic chunking, hybrid retrieval, knowledge graphs, and LLM-as-Judge evaluation.**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [API Reference](#api-reference) • [Concepts](#core-concepts--techniques)

</div>

---

## Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Architecture](#-architecture)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Core Concepts & Techniques](#-core-concepts--techniques)
   - [Document Processing](#1-document-processing)
   - [Semantic Chunking](#2-semantic-chunking)
   - [Embedding Generation](#3-embedding-generation)
   - [Vector Store](#4-vector-store)
   - [Hybrid Retrieval](#5-hybrid-retrieval)
   - [Query Enhancement](#6-query-enhancement)
   - [Cross-Encoder Reranking](#7-cross-encoder-reranking)
   - [Query Routing](#8-query-routing)
   - [Knowledge Graph](#9-knowledge-graph)
   - [RAG Chain & Generation](#10-rag-chain--generation)
   - [ReAct Orchestration](#11-react-orchestration)
   - [Graph-Based Agentic RAG](#12-graph-based-agentic-rag-ga-rag)
   - [LLM-as-Judge Evaluation](#13-llm-as-judge-evaluation)
   - [Mathematical Foundations](#14-mathematical-foundations)
7. [API Reference](#-api-reference)
8. [Configuration](#-configuration)
9. [Testing](#-testing)
10. [Contributing](#-contributing)

---

## Overview

This system is a comprehensive **Retrieval-Augmented Generation (RAG)** pipeline specifically designed for **precision agriculture** and **water management** domains. It processes technical documents (PDFs), extracts knowledge, and provides accurate, citation-backed answers to agricultural queries.

### Key Differentiators

- **Domain-Specific**: Tailored for irrigation, soil moisture, crop management, and water conservation
- **Morocco-Focused**: Special support for Moroccan regions, water basins, and local agricultural contexts
- **Multi-Modal Retrieval**: Combines semantic search, keyword matching, and knowledge graph traversal
- **Agentic Architecture**: Autonomous query decomposition and iterative refinement
- **Mathematically Grounded**: Implements research-backed metrics for chunking quality and retrieval evaluation

### System Specifications

| Component | Technology |
|-----------|------------|
| **LLM** | Llama 3.2 3B via Ollama |
| **Embeddings** | all-MiniLM-L6-v2 (384 dims) |
| **Vector Store** | ChromaDB (HNSW indexing) |
| **Knowledge Graph** | NetworkX |
| **API** | FastAPI with Swagger |
| **Optimized For** | i5-10300H, 16GB RAM, GTX 1650 (4GB VRAM) |

---

## Features

### Document Processing
- ✅ PDF extraction with table handling (pdfplumber + pypdf fallback)
- ✅ Semantic chunking with topic boundary detection
- ✅ Propositional decomposition (atomic fact extraction)
- ✅ Morocco-specific metadata tagging (basins, regions)

### Retrieval
- ✅ **Hybrid Search**: Semantic embeddings + BM25 keyword matching
- ✅ **Multi-Query Generation**: LLM-generated query variations
- ✅ **Cross-Encoder Reranking**: Precision re-scoring of candidates
- ✅ **Smart Query Routing**: Intent-based retrieval strategy selection
- ✅ **Sentence-Window Retrieval**: Fine-grained retrieval with context expansion

### Knowledge Graph
- ✅ Entity extraction (SpaCy NER + agricultural patterns)
- ✅ Relationship extraction via co-occurrence and dependency parsing
- ✅ Multi-hop graph traversal for relational queries
- ✅ NetworkX-based in-memory graph storage

### Generation
- ✅ Ollama integration (supports Llama 3, Qwen, Gemma, etc.)
- ✅ Conversation memory with sliding window
- ✅ Structured output with citations
- ✅ Confidence scoring

### Advanced Features
- ✅ **ReAct Orchestration**: Thought → Action → Observation → Critique loop
- ✅ **GA-RAG Agent**: Query decomposition + tool selection + iterative synthesis
- ✅ **LLM-as-Judge Evaluation**: Faithfulness, relevancy, and agricultural accuracy metrics
- ✅ **FastAPI REST API** with Swagger documentation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRECISION AGRICULTURE RAG                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   FastAPI   │  │     CLI     │  │  Notebook   │  │   Query Interface   │ │
│  │   Server    │  │  Interface  │  │  Interface  │  │     (Interactive)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                    │            │
│         └────────────────┴────────────────┴────────────────────┘            │
│                                    │                                         │
│                           ┌────────▼────────┐                               │
│                           │  RAG Orchestrator│                               │
│                           │   (ReAct Loop)   │                               │
│                           └────────┬────────┘                               │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         │                          │                          │             │
│  ┌──────▼──────┐           ┌───────▼───────┐          ┌───────▼───────┐    │
│  │   Query     │           │    Hybrid     │          │   Knowledge   │    │
│  │   Router    │           │   Retriever   │          │     Graph     │    │
│  │  (Intent)   │           │ (Semantic+BM25)│          │   (NetworkX)  │    │
│  └─────────────┘           └───────┬───────┘          └───────────────┘    │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│             ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐                │
│             │   Query     │ │   Cross-    │ │  Sentence   │                │
│             │  Enhancer   │ │  Encoder    │ │   Window    │                │
│             │ (Multi-Q)   │ │  Reranker   │ │  Retriever  │                │
│             └─────────────┘ └─────────────┘ └─────────────┘                │
│                                    │                                         │
│                           ┌────────▼────────┐                               │
│                           │   RAG Chain     │                               │
│                           │   (LangChain)   │                               │
│                           └────────┬────────┘                               │
│                                    │                                         │
│                           ┌────────▼────────┐                               │
│                           │     Ollama      │                               │
│                           │  (Llama 3/Qwen) │                               │
│                           └─────────────────┘                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA LAYER                                  │   │
│  ├─────────────┬──────────────┬─────────────────┬─────────────────────┤   │
│  │  ChromaDB   │  Knowledge   │    Semantic     │   Embedding Cache   │   │
│  │ VectorStore │    Graph     │     Chunks      │      (Pickle)       │   │
│  └─────────────┴──────────────┴─────────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
├── src/
│   ├── adapters/                 # Interface adapters
│   │   ├── api/                  # FastAPI REST server
│   │   └── cli/                  # Command-line interface
│   ├── application/              # Application layer
│   │   ├── agents/               # GA-RAG agent implementation
│   │   └── evaluation/           # LLM-as-Judge evaluators
│   ├── config/                   # Configuration management
│   ├── core/                     # Core entities and interfaces
│   ├── document_processing/      # PDF processing & chunking
│   ├── embeddings/               # Embedding generation & vector store
│   ├── generation/               # LLM generation & memory
│   ├── infrastructure/           # Infrastructure (LLM clients, persistence)
│   ├── knowledge_graph/          # Entity extraction & graph building
│   ├── retrieval/                # Hybrid retrieval pipeline
│   └── utils/                    # Utility functions
├── data/                         # Generated data
│   ├── vector_store/             # ChromaDB persistent storage
│   ├── knowledge_graph/          # Graph serialization
│   ├── processed_chunks/         # Chunked document data
│   └── embedding_cache/          # Cached embeddings
├── documents/                    # Source PDF documents
├── tests/                        # Unit & integration tests
├── notebooks/                    # Jupyter notebooks
└── logs/                         # Application logs
```

---

## Installation

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running
- 16GB RAM recommended (4GB VRAM for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/precision-agriculture-rag.git
cd precision-agriculture-rag
```

2. **Run setup script**
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Pull LLM model with Ollama**
```bash
ollama pull llama3.2:3b
# Or for better quality:
ollama pull gemma2:9b
```

4. **Add documents**
Place PDF files in the `documents/` folder, organized by category:
```
documents/
├── 01_core_water_agriculture/
├── 02_precision_agriculture_tech/
├── 03_irrigation_systems/
└── ...
```

5. **Ingest documents**
```bash
python src/ingest.py
```

---

## Usage

### Interactive CLI

```bash
python src/main.py
```

Commands:
- `/react` - Use ReAct reasoning (structured output)
- `/eval` - Evaluate last response
- `/clear` - Clear conversation memory
- `/quit` - Exit

### Single Query

```bash
python src/main.py --query "What are the best practices for drip irrigation?"
```

### REST API

```bash
./start_api.sh
# Or:
uvicorn src.adapters.api.server:app --reload
```

API Endpoints:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- `POST /query` - Query the RAG system
- `GET /health` - Health check
- `GET /info` - System information

### Python SDK

```python
from src.main import PrecisionAgricultureRAG

rag = PrecisionAgricultureRAG()

response = rag.query(
    question="How does soil moisture affect irrigation scheduling?",
    top_k=5,
    use_reranking=True
)

print(response["answer"])
print(f"Confidence: {response['confidence']:.0%}")
```

---

## Core Concepts & Techniques

### 1. Document Processing

**Module**: `src/document_processing/processor.py`

The document processor handles PDF extraction with intelligent table detection and text cleaning.

#### Features
- **Dual extraction engines**: pdfplumber (primary) with pypdf fallback
- **Table detection**: Structured table data converted to text format
- **Text normalization**: Quote normalization, whitespace cleanup, header/footer removal

#### Pipeline
```
PDF → Page Extraction → Table Detection → Text Cleaning → Chunk Generation
```

#### Metadata Extraction
Each chunk includes:
- Source document name
- Category (from folder structure)
- Page number
- Chunk index
- Topic tags (domain-specific keywords)

---

### 2. Semantic Chunking

**Module**: `src/document_processing/semantic_chunker.py`

Implements **SemanticContextAgent v2.0** - an agentic proposition-based semantic splitter.

#### Configuration
```python
SEMANTIC_CHUNK_CONFIG = {
    "target_token_range": [200, 500],      # Target chunk size in tokens
    "min_chunk_chars": 800,                 # ~200 tokens
    "max_chunk_chars": 2000,                # ~500 tokens
    "overlap_percent": 0.10,                # 10% semantic bridge
    "breakpoint_threshold_percentile": 95,  # For semantic breaking
    "similarity_drop_threshold": 0.15,      # Topic shift detection
}
```

#### Algorithm: Semantic Similarity Change Point Detection

The chunker uses **embedding-based topic boundary detection**:

1. **Propositional Split**: Text → Sentences (atomic units)
2. **Embedding Generation**: Each sentence → 384-dim vector
3. **Similarity Calculation**: Consecutive sentence cosine similarity
4. **Breakpoint Detection**: Identify where dissimilarity > threshold

**Mathematical Formula:**

```
breakpoint_i = 1  if (1 - sim(s_i, s_{i+1})) >= percentile_95
             = 0  otherwise
```

Where cosine similarity:
```
sim(s_i, s_{i+1}) = (e_i · e_{i+1}) / (||e_i|| × ||e_{i+1}||)
```

#### Sliding Window Change Point Detection

Uses a sliding window (W=3) with statistical threshold:

```
Boundary Trigger: sim(s_i, s_{i+1}) < μ - (k × σ)
```

Where:
- μ = local mean similarity in window
- σ = local standard deviation
- k ∈ [1.5, 2.0] = threshold multiplier

#### Features Implemented

| Feature | Description |
|---------|-------------|
| **Contextual Chunk Headers** | `[Document: X] [Section: Y] [Page: Z]` prepended to each chunk |
| **Sliding Windows** | Configurable window/stride for overlapping views |
| **Hierarchical Chunking** | Document → Section → Chunk → Sentence levels |
| **Sentence-Window Retrieval** | Store N sentences before/after for context expansion |
| **Intent Classification** | Technical, Instructional, or Conceptual labeling |

#### Chunking Quality Metrics

The `ChunkingEvaluator` class implements mathematical evaluation:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Boundary Clarity** | ∇sim_i = sim(S_i, S_{i+1}) - sim(S_{i-1}, S_i) | Gradient of cosine similarity |
| **Chunk Stickiness** | σ²_inter / σ²_intra | Inter-chunk vs intra-chunk variance ratio |
| **CIG Score** | H(chunk) - Σ MI(chunk, others) | Contextual Information Gain |

**Stickiness Formula:**
```
Stickiness = Var(inter-chunk similarities) / Var(intra-chunk similarities)
```

Higher stickiness = better chunk cohesion.

---

### 3. Embedding Generation

**Module**: `src/embeddings/generator.py`

#### Model: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Speed**: Optimized for real-time inference
- **Normalization**: L2 normalized for cosine similarity

#### Features
- **GPU Acceleration**: CUDA support when available
- **Batch Processing**: Configurable batch size (default: 32)
- **Caching**: Disk-based embedding cache using MD5 hashing

```python
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda",  # or "cpu"
    "batch_size": 32,
    "normalize_embeddings": True,
}
```

#### Similarity Computation

For normalized embeddings, **dot product = cosine similarity**:

```
similarity(q, d) = q · d = cos(θ)
```

---

### 4. Vector Store

**Module**: `src/embeddings/vector_store.py`

#### ChromaDB Configuration
```python
VECTOR_STORE_CONFIG = {
    "type": "chroma",
    "collection_name": "water_management_docs",
    "distance_metric": "cosine",
    "embedding_dimension": 384,
}
```

#### HNSW Index

ChromaDB uses **Hierarchical Navigable Small World (HNSW)** graphs for approximate nearest neighbor search:

- **Time Complexity**: O(log N) per query
- **Space Complexity**: O(N × M) where M = connections per node

---

### 5. Hybrid Retrieval

**Module**: `src/retrieval/hybrid_retriever.py`

Combines **semantic search** (embeddings) with **keyword search** (BM25) using fusion algorithms.

#### BM25 (Best Matching 25)

BM25 scoring for keyword search:

```
BM25(D, Q) = Σ IDF(q_i) × [f(q_i, D) × (k₁ + 1)] / [f(q_i, D) + k₁ × (1 - b + b × |D|/avgdl)]
```

Where:
- f(q_i, D) = term frequency in document
- |D| = document length
- avgdl = average document length
- k₁ = 1.2 (term frequency saturation)
- b = 0.75 (length normalization)

#### Inverse Document Frequency:
```
IDF(q_i) = ln[(N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1]
```

#### Reciprocal Rank Fusion (RRF)

Combines multiple ranking lists:

```
RRF_score(d) = Σ 1 / (k + rank_r(d))
```

Where:
- R = set of rankings (semantic, keyword)
- k = 60 (constant to prevent high-ranked dominance)
- rank_r(d) = position of document d in ranking r

#### Weighted Fusion

Alternative fusion using normalized scores:

```
combined(d) = α × [semantic(d)/max(semantic)] + (1-α) × [keyword(d)/max(keyword)]
```

Default: α = 0.7 (70% semantic, 30% keyword)

---

### 6. Query Enhancement

**Module**: `src/retrieval/query_enhancer.py`

#### Multi-Query Generation

Uses LLM to generate diverse query reformulations:

```
Original: "How to save water in irrigation?"
→ Query 1: "What are water conservation techniques for irrigation systems?"
→ Query 2: "Methods to reduce water usage in agricultural irrigation"
→ Query 3: "Efficient irrigation practices for water savings"
```

#### Domain-Specific Query Expansion

Adds synonyms from agricultural vocabulary:

```python
DOMAIN_SYNONYMS = {
    "irrigation": ["watering", "water application"],
    "drip": ["trickle", "micro-irrigation"],
    "efficiency": ["productivity", "optimization"],
    "evapotranspiration": ["ET", "ET0", "water loss"],
}
```

---

### 7. Cross-Encoder Reranking

**Module**: `src/retrieval/reranker.py`

#### Model: cross-encoder/ms-marco-MiniLM-L-6-v2

Cross-encoders process query-document pairs jointly, providing more accurate relevance scores than bi-encoders.

#### Architecture Comparison

| Bi-Encoder | Cross-Encoder |
|------------|---------------|
| Query → Embedding | (Query, Document) → Score |
| Document → Embedding | Joint attention |
| Dot product similarity | Classification head |
| Fast (separate encoding) | Slow (pair-wise) |
| Used for recall | Used for precision |

#### Two-Stage Retrieval

```
Stage 1: Bi-encoder retrieves top-20 candidates (fast)
Stage 2: Cross-encoder reranks to top-5 (precise)
```

---

### 8. Query Routing

**Module**: `src/retrieval/query_router.py`

Intelligent classification of query intent to select optimal retrieval strategy.

#### Query Intents

| Intent | Pattern | Strategy |
|--------|---------|----------|
| FACTUAL | "What is...", "Define..." | Vector search, high semantic weight |
| RELATIONAL | "How are X and Y related?" | Graph traversal, multi-hop |
| SUMMARY | "Summarize...", "Overview of..." | Vector search, more documents |
| REASONING | "Why...", "Analyze..." | Hybrid + Graph, 2-hop depth |
| PROCEDURAL | "How to...", "Steps to..." | Vector + Keyword |
| COMPARATIVE | "Compare X and Y" | Hybrid + Graph |

#### Retrieval Strategy Configuration

```python
@dataclass
class RetrievalStrategy:
    use_vector: bool = True
    use_keyword: bool = True
    use_graph: bool = False
    graph_depth: int = 1
    top_k: int = 5
    semantic_weight: float = 0.7
    use_reranking: bool = True
```

---

### 9. Knowledge Graph

**Module**: `src/knowledge_graph/graph_builder.py`

#### Entity Types

| Type | Examples |
|------|----------|
| irrigation_method | drip irrigation, sprinkler, center pivot |
| crop | wheat, tomato, olive, citrus |
| sensor | soil moisture sensor, tensiometer, TDR |
| technology | precision agriculture, IoT, remote sensing |
| water_source | groundwater, reservoir, aquifer |
| metric | evapotranspiration, water use efficiency |
| region | arid, Mediterranean, Morocco basins |
| organization | FAO, USDA, ICARDA |

#### Entity Extraction

1. **Pattern-Based**: Agricultural domain regex patterns
2. **SpaCy NER**: Named entity recognition for organizations, locations
3. **Hybrid**: Combines both methods with deduplication

#### Relationship Extraction

- **Co-occurrence**: Entities in same sentence
- **Dependency Parsing**: Subject-verb-object triples
- **Proximity**: Entities within 500 characters

#### Graph Queries

**Ego Graph Retrieval:**
```python
def query_entity(entity: str, max_depth: int = 2):
    subgraph = nx.ego_graph(self.graph, entity, radius=max_depth)
    # Returns neighbors, predecessors, relationships
```

#### Graph Metrics

- **Nodes**: Unique entities
- **Edges**: Entity relationships
- **Connected Components**: Isolated subgraphs
- **Average Degree**: Relationship density

---

### 10. RAG Chain & Generation

**Module**: `src/generation/rag_chain.py`

#### Prompt Engineering

Uses Llama 3 chat format with structured sections:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{conversation_history}
RETRIEVED CONTEXT:
{context}
KNOWLEDGE GRAPH CONTEXT:
{kg_context}
QUESTION: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

#### System Prompt

Specialized for agricultural advisory:
- Irrigation systems and scheduling
- Soil moisture management
- Water conservation techniques
- IoT sensors and monitoring
- Crop-specific water requirements

#### Confidence Scoring

Heuristic-based confidence estimation:

```python
confidence = 0.5  # Base

# More sources → higher confidence
if num_docs >= 3:
    confidence += 0.2

# Citations in response → higher confidence
if "[Source:" in response:
    confidence += 0.1

# Uncertainty phrases → lower confidence
if "not sure" in response.lower():
    confidence -= 0.15
```

#### Conversation Memory

Sliding window memory (last 5 exchanges):

```python
MEMORY_CONFIG = {
    "type": "conversation_buffer_window",
    "k": 5,
    "memory_key": "chat_history",
}
```

---

### 11. ReAct Orchestration

**Module**: `src/generation/rag_orchestrator.py`

Implements **ReAct (Reasoning + Acting)** pattern for multi-step reasoning.

#### ReAct Loop

```
1. THOUGHT: Analyze what information is needed
2. ACTION: Choose and call a retrieval tool
3. OBSERVATION: Analyze the results
4. CRITIQUE: Decide if more information is needed
5. (Repeat until sufficient context gathered)
6. SYNTHESIZE: Generate final answer
```

#### Available Tools

| Tool | Description | Best For |
|------|-------------|----------|
| `vector_search(query, top_k)` | Semantic similarity search | Facts, definitions |
| `graph_search(entities, depth)` | Multi-hop entity traversal | Relationships, hierarchies |
| `keyword_search(query)` | BM25 exact matching | Technical terms, IDs |
| `get_document_summary(doc_id)` | Document overview | Big-picture context |

#### Structured Output Format

```markdown
## Executive Summary
Brief 2-3 sentence overview

## Detailed Findings
1. Finding with [Source: document_name]
2. Another finding with citation

## Sources
| # | Document | Relevance |
|---|----------|-----------|
| 1 | doc.pdf  | High      |
```

---

### 12. Graph-Based Agentic RAG (GA-RAG)

**Module**: `src/application/agents/rag_agent.py`

Advanced agentic architecture with autonomous decision-making.

#### Three Phases

**Phase 1: Query Decomposition**
- Break complex queries into sub-problems
- Classify each as: FACTUAL, RELATIONAL, COMPARATIVE, PROCEDURAL
- Extract entities for graph traversal

**Phase 2: Tool Selection**
- Route sub-queries to optimal retrieval tool
- RELATIONAL → Graph search
- FACTUAL → Vector search
- Mixed → Hybrid approach

**Phase 3: Iterative Synthesis**
- Execute tools in sequence
- Accumulate context
- Refine until quality threshold met

#### Stateful Memory (Loop Prevention)

```python
@dataclass
class AgentState:
    visited_entities: Set[str]      # Prevent re-visiting
    traversal_paths: List[List[str]]  # Track exploration
    retrieval_attempts: int         # Max iterations guard
    max_attempts: int = 5           # Hard limit
    context_buffer: List[str]       # Accumulated context
```

#### Query Decomposition Example

```
Input: "How does drip irrigation affect water efficiency, and 
        what sensors are used to monitor soil moisture?"

Decomposition:
1. FACTUAL: "What is drip irrigation?" | ENTITIES: drip irrigation
2. RELATIONAL: "How does drip irrigation affect water efficiency?" 
   | ENTITIES: drip irrigation, water efficiency
3. FACTUAL: "What sensors monitor soil moisture?" 
   | ENTITIES: sensors, soil moisture
```

---

### 13. LLM-as-Judge Evaluation

**Modules**: `src/application/evaluation/rag_eval.py`, `src/utils/evaluator.py`

Implements multiple evaluation judges for RAG quality assessment.

#### Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Faithfulness** | Claims supported by context | supported_claims / total_claims |
| **Answer Relevancy** | Response addresses question | LLM judgment [0-10] |
| **Context Relevancy** | Retrieved context is useful | LLM judgment [0-10] |
| **Context Precision** | Relevant docs ranked higher | AP@k |
| **Context Recall** | All relevant info retrieved | Coverage assessment |

#### Agricultural-Specific Judges

**AgriculturalAccuracyJudge:**
- Numerical accuracy (measurements, units)
- Technical term correctness
- Practical applicability for farmers
- Regional appropriateness

**SafetyJudge:**
- No incorrect irrigation volumes
- No wrong chemical concentrations
- No soil degradation risks
- Appropriate caveats included

#### Verdict Types

```python
class VerdictType(Enum):
    PASS = "PASS"       # Score ≥ 0.7
    MARGINAL = "MARGINAL"  # 0.4 ≤ Score < 0.7
    FAIL = "FAIL"       # Score < 0.4
```

#### Evaluation Report Example

```json
{
  "scores": {
    "faithfulness": {"score": 0.85, "reason": "All claims supported"},
    "answer_relevancy": {"score": 0.9, "reason": "Direct answer"},
    "agricultural_accuracy": {"score": 0.8, "reason": "Correct metrics"}
  },
  "overall_score": 0.85,
  "verdict": "PASS",
  "recommendations": []
}
```

---

### 14. Mathematical Foundations

#### Cosine Similarity

Core similarity measure for embedding comparison:

```
cos(u, v) = (u · v) / (||u|| × ||v||) = Σ(u_i × v_i) / (√Σu_i² × √Σv_i²)
```

#### Normalized Dot Product

For L2-normalized embeddings:
```
if ||u|| = ||v|| = 1:  cos(u, v) = u · v
```

#### Euclidean Distance to Cosine

For normalized vectors:
```
||u - v||² = 2(1 - cos(u, v))
```

#### Information Entropy

Used in CIG (Contextual Information Gain) calculation:

```
H(X) = -Σ p(x_i) × log(p(x_i))
```

Approximated via vocabulary diversity:
```
H_hat(chunk) = |unique_words| / |total_words| × min(1, |words|/200)
```

#### Mutual Information

For measuring redundancy between chunks:

```
MI(X; Y) ≈ cosine_similarity(X, Y)
```

High MI = High redundancy (bad for chunk diversity)

#### Variance Calculations

**Intra-chunk variance** (should be low for cohesion):
```
σ²_intra = (1/|C|) × Σ Var({sim(s_i, s_j) : s_i, s_j ∈ c})
```

**Inter-chunk variance** (should be high for separation):
```
σ²_inter = Var({sim(c_i, c_{i+1}) : i = 1, ..., n-1})
```

#### Gradient for Boundary Detection

```
∇sim_i = sim(S_i, S_{i+1}) - sim(S_{i-1}, S_i)
```

Negative gradient indicates topic shift (good boundary location).

---

## API Reference

### REST Endpoints

#### POST /query
Query the RAG system.

**Request:**
```json
{
  "question": "What are the best practices for drip irrigation?",
  "top_k": 5,
  "use_reranking": true,
  "use_agent": false
}
```

**Response:**
```json
{
  "answer": "Drip irrigation best practices include...",
  "sources": ["irrigation_guide.pdf", "water_management.pdf"],
  "confidence": 0.85,
  "query_time_ms": 1234.5,
  "tools_used": ["vector_search", "reranker"]
}
```

#### GET /health
Check system health.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "rag_system": true,
    "vector_store": true,
    "knowledge_graph": true
  }
}
```

#### GET /info
Get system information.

**Response:**
```json
{
  "vector_store_count": 1523,
  "knowledge_graph_nodes": 456,
  "knowledge_graph_edges": 892,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "llama3.2:3b"
}
```

---

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# LLM Configuration
LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.3

# Embedding Configuration
EMBEDDING_DEVICE=cuda  # or cpu

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `src/config/config.py` | Central configuration |
| `pytest.ini` | Test configuration |
| `requirements.txt` | Python dependencies |

### LLM Configuration

```python
LLM_CONFIG = {
    "model": "llama3.2:3b",
    "base_url": "http://localhost:11434",
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "num_ctx": 4096,
    "repeat_penalty": 1.2,
    "num_predict": 512,
}
```

### Retrieval Configuration

```python
RETRIEVAL_CONFIG = {
    "hybrid_search": {
        "enabled": True,
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "top_k_semantic": 20,
        "top_k_keyword": 10,
    },
    "multi_query": {
        "enabled": True,
        "num_queries": 3,
        "aggregation_method": "reciprocal_rank_fusion",
    },
    "reranking": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k_initial": 20,
        "top_k_final": 5,
    },
    "final_top_k": 5,
}
```

---

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_core.py::TestChunkingEvaluator

# Verbose output
pytest -v tests/
```

### Test Categories

| Module | Tests |
|--------|-------|
| `test_core.py` | ChunkingEvaluator, RAGAgent, KnowledgeGraph |
| `evaluate_rag.py` | End-to-end RAG evaluation |

### Test Dataset

The system includes an irrigation Q&A dataset (`tests/irrigation_qa_dataset.json`) for evaluation.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions
- Add unit tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **SentenceTransformers** - Embedding models
- **Ollama** - Local LLM inference
- **SpaCy** - NLP and NER
- **NetworkX** - Graph algorithms
- **FastAPI** - REST API framework
- **rank-bm25** - BM25 implementation

---

## References

### Papers & Methodologies

1. **RAG (Retrieval-Augmented Generation)**: Lewis et al., 2020
2. **BM25**: Robertson & Zaragoza, 2009
3. **Sentence-BERT**: Reimers & Gurevych, 2019
4. **Cross-Encoders**: Nogueira & Cho, 2019
5. **ReAct**: Yao et al., 2022
6. **HNSW**: Malkov & Yashunin, 2018

### Domain Knowledge

- FAO Water Management Guidelines
- ICARDA Irrigation Best Practices
- Morocco National Water Strategy

---

<div align="center">
  <sub>Built for Precision Agriculture</sub>
  <br>
  <sub>Optimized for Morocco Water Management</sub>
</div>
