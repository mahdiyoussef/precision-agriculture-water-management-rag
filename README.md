# Precision Agriculture Water Management RAG System

An advanced **Retrieval-Augmented Generation (RAG)** system for precision agriculture and water management, featuring **Graph-Based Agentic RAG (GA-RAG)**, **semantic chunking evaluation**, and **LLM-as-Judge evaluation framework**.

## System Overview

| Component | Technology |
|-----------|------------|
| **LLM** | Llama 3.2 3B via Ollama |
| **Embeddings** | all-MiniLM-L6-v2 (384 dims) |
| **Vector Store** | ChromaDB |
| **Graph** | NetworkX |
| **Hardware** | i5-10300H, 16GB RAM, GTX 1650 (4GB VRAM) |

## Key Features

### Retrieval
- **Hybrid Search** - Semantic (embeddings) + Keyword (BM25) with Reciprocal Rank Fusion
- **Cross-Encoder Reranking** - ms-marco-MiniLM-L-6-v2 for precision scoring
- **Query Enhancement** - Multi-query generation and domain-specific expansion
- **Intelligent Routing** - Auto-classifies queries (factual/relational/comparative/procedural)

### Agentic RAG (GA-RAG)
- **Query Decomposition** - Breaks complex queries into sub-problems
- **Tool Selection** - Routes to Vector Search or Graph Traversal
- **Iterative Synthesis** - Self-refining context retrieval
- **Stateful Memory** - Prevents infinite loops during multi-hop reasoning

### Knowledge Graph
- **Entity Extraction** - SpaCy NER + domain-specific patterns
- **BFS Traversal** - Multi-hop relational reasoning
- **PageRank Scoring** - Entity importance ranking
- **Shortest Path** - Relationship discovery between concepts

### Evaluation Framework
- **RAG Triad** - Faithfulness, Answer Relevancy, Context Relevancy
- **Chunking Metrics** - Boundary Clarity, Chunk Stickiness, CIG
- **Agentic Metrics** - Step Efficiency, PPI Calibration
- **LLM-as-Judge** - Automated evaluation with human calibration

---

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** with LLM model:
   ```bash
   # Install from https://ollama.ai
   ollama pull llama3.2:3b
   ```

### Installation

```bash
# Clone and setup
git clone <repository>
cd "RAG For Precision Agriculture and Water management"

# Create environment
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage

```bash
source rag_env/bin/activate

# Ingest documents
python src/ingest.py

# Interactive chat
python src/main.py

# Single query
python src/main.py --query "Best practices for drip irrigation?"
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/react` | Use ReAct orchestrator |
| `/eval` | Evaluate last response |
| `/agent` | Use Agentic RAG |
| `/info` | System statistics |
| `/clear` | Clear history |
| `/quit` | Exit |

---

## Architecture

```
src/
+-- core/                    # Domain Layer
|   +-- entities/            # SemanticChunk, QueryType, AgenticMetrics
|   +-- interfaces/          # RetrieverProtocol, LLMProtocol
|   +-- exceptions.py        # Custom exceptions
|
+-- application/             # Use Cases
|   +-- agents/              # RAGAgent, RAGOrchestrator
|   +-- evaluation/          # LLM-as-Judge, PPI Calibration
|
+-- document_processing/     # Chunking
|   +-- semantic_chunker.py  # Proposition-based semantic splitting
|
+-- embeddings/              # Vector operations
|   +-- generator.py         # SentenceTransformer
|   +-- vector_store.py      # ChromaDB wrapper
|
+-- retrieval/               # Search
|   +-- hybrid_retriever.py  # Semantic + BM25
|   +-- query_router.py      # Intent classification
|   +-- query_enhancer.py    # Multi-query expansion
|   +-- reranker.py          # Cross-encoder
|   +-- rag_agent.py         # GA-RAG implementation
|
+-- knowledge_graph/         # Graph operations
|   +-- graph_builder.py     # Entity/relationship extraction
|
+-- generation/              # Response generation
|   +-- rag_chain.py         # LangChain pipeline
|   +-- rag_orchestrator.py  # ReAct reasoning
|   +-- memory_manager.py    # Conversation history
|
+-- config/                  # Configuration
    +-- config.py            # All settings
```

---

## Document Corpus

| Category | Documents | Topics |
|----------|-----------|--------|
| Core Water Agriculture | 9 | FAO guidelines, water management fundamentals |
| Precision Agriculture Tech | 11 | IoT, sensors, remote sensing, AI |
| Irrigation Systems | 8 | Drip, sprinkler, surface irrigation |
| Water Resources | 7 | Groundwater, aquifer management |
| Rainfed & Smallscale | 3 | Traditional methods, smallholder farming |
| Specific Sectors | 3 | Cotton, citrus, specialized crops |
| Policy & Strategy | 7 | National water strategies, Morocco focus |

---

## Configuration

Key settings in `src/config/config.py`:

```python
# LLM
LLM_CONFIG = {
    "model": "llama3.2:3b",
    "temperature": 0.3,
    "num_ctx": 4096,
}

# Retrieval
RETRIEVAL_CONFIG = {
    "hybrid_search": {"semantic_weight": 0.7, "keyword_weight": 0.3},
    "reranking": {"top_k_initial": 20, "top_k_final": 5},
}

# Chunking
CHUNK_CONFIG = {
    "chunk_size": 600,
    "chunk_overlap_percent": 0.15,
    "breakpoint_threshold_percentile": 95,
}
```

---

## Evaluation

### Run Evaluation
```bash
python tests/evaluate_rag.py --num 10
```

### Metrics
- **Faithfulness**: Claims grounded in context (0-1)
- **Answer Relevancy**: Response addresses query (0-1)
- **Context Relevancy**: Retrieved docs are relevant (0-1)
- **Agricultural Accuracy**: Domain correctness (0-1)
- **Step Efficiency**: shortest_path / actual_hops (0-1)

### PPI Calibration
Calibrate LLM-as-Judge against human baseline:
```python
from src.application.evaluation import AgenticEvaluator

evaluator = AgenticEvaluator()
evaluator.calibrate_with_baseline(human_scores, llm_scores)
```

---

## API Usage

### Basic Query
```python
from src.main import PrecisionAgricultureRAG

rag = PrecisionAgricultureRAG()
result = rag.query("How does drip irrigation affect water efficiency?")
print(result["answer"])
```

### Agentic Query
```python
from src.application.agents import RAGAgent

agent = RAGAgent(
    hybrid_retriever=rag.hybrid_retriever,
    knowledge_graph=rag.knowledge_graph
)
result = agent.run("Compare irrigation methods for arid regions")
print(f"Answer: {result.answer}")
print(f"Tools: {result.tools_used}")
print(f"Hops: {result.hops_taken}")
```

### Chunking Evaluation
```python
from src.document_processing.semantic_chunker import SemanticChunker, ChunkingEvaluator

chunker = SemanticChunker()
evaluator = ChunkingEvaluator(embedding_model=chunker.embedding_model)

chunks = chunker.process_document("path/to/doc.pdf")
metrics = evaluator.evaluate_chunks(chunks)
print(metrics.summary())
```

---

## Technical Details

### Retrieval Pipeline
1. **Query Classification** - Intent detection (6 types)
2. **Query Enhancement** - 3 query variations via LLM
3. **Hybrid Search** - ChromaDB + BM25
4. **Fusion** - Reciprocal Rank Fusion
5. **Reranking** - Cross-encoder top-20 -> top-5
6. **Graph Context** - Entity relationship enrichment

### Semantic Chunking
- **Proposition Splitting** - Sentence-level decomposition
- **Semantic Clustering** - Cosine similarity grouping
- **Sliding Window** - W=3, sigma threshold for boundaries
- **Contextual Headers** - Document hierarchy injection

### Knowledge Graph
- **Entity Types**: crop, irrigation_method, sensor, technology, region, water_source, metric
- **Algorithms**: BFS traversal, PageRank, shortest path

---

## Example Queries

- "What are best practices for drip irrigation in arid regions?"
- "How do soil moisture sensors improve water efficiency?"
- "Compare sprinkler vs drip irrigation for cotton crops"
- "What is evapotranspiration and how is it measured?"
- "How does precision agriculture reduce water waste?"

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Follow Clean Architecture patterns
4. Add tests for new features
5. Submit pull request
