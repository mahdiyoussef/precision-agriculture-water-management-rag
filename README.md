# Precision Agriculture Water Management RAG System

An advanced **Retrieval-Augmented Generation (RAG)** system for precision agriculture and water management, featuring **Graph-Based Agentic RAG (GA-RAG)**, **semantic chunking evaluation**, and **LLM-as-Judge evaluation framework**.

## System Overview

| Component | Technology |
|-----------|------------|
| **LLM** | Llama 3.2 3B via Ollama |
| **Embeddings** | all-MiniLM-L6-v2 (384 dims) |
| **Vector Store** | ChromaDB |
| **Graph** | NetworkX |
| **API** | FastAPI with Swagger |
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

### REST API
- **FastAPI Server** - Production-ready REST endpoints
- **Swagger UI** - Interactive API documentation at `/docs`
- **Streaming** - Server-Sent Events for real-time responses
- **Evaluation Endpoint** - LLM-as-Judge via API

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

# Optional: Copy environment config
cp .env.example .env
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

### REST API Server

```bash
# Start API server (Swagger at http://localhost:8000/docs)
./start_api.sh

# Or with custom port
./start_api.sh 8080
```

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | System info |
| `/query` | POST | Query RAG system |
| `/query/stream` | POST | Streaming response (SSE) |
| `/evaluate` | POST | LLM-as-Judge evaluation |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

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

## Environment Configuration

Copy `.env.example` to `.env` and customize:

```bash
# LLM Configuration
LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.3

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda

# Performance
ENABLE_CACHING=true
```

---

## Architecture

```
src/
+-- adapters/                # Entry Points
|   +-- api/                 # FastAPI REST server
|   +-- cli/                 # Command-line interface
|
+-- core/                    # Domain Layer
|   +-- entities/            # SemanticChunk, QueryType, AgenticMetrics
|   +-- interfaces/          # RetrieverProtocol, LLMProtocol
|   +-- exceptions.py        # Custom exceptions
|
+-- application/             # Use Cases
|   +-- agents/              # RAGAgent, RAGOrchestrator
|   +-- evaluation/          # LLM-as-Judge, PPI Calibration
|
+-- infrastructure/          # Shared Resources
|   +-- shared.py            # Singleton registries, caching
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
    +-- config.py            # All settings (supports .env)
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

## API Usage

### Python SDK
```python
from src.main import PrecisionAgricultureRAG

rag = PrecisionAgricultureRAG()
result = rag.query("How does drip irrigation affect water efficiency?")
print(result["answer"])
```

### REST API
```bash
# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is drip irrigation?", "top_k": 5}'

# Agentic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare irrigation methods", "use_agent": true}'
```

### Agentic Query (Python)
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

---

## Testing

```bash
# Run unit tests
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ -v --cov=src

# Run evaluation
python tests/evaluate_rag.py --num 10
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
