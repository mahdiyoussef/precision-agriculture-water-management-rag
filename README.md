# Precision Agriculture Water Management RAG System

An advanced Retrieval-Augmented Generation (RAG) system for precision agriculture and water management, optimized for:
- **Hardware**: Intel i5-10300H, 16GB RAM, NVIDIA GTX 1650 (4GB VRAM)
- **LLM**: Qwen 1.8B via Ollama
- **Framework**: LangChain

## Features

- **Hybrid Retrieval**: Combines semantic search (embeddings) with keyword search (BM25)
- **Query Enhancement**: Multi-query generation and domain-specific expansion
- **Cross-Encoder Reranking**: Precise relevance scoring for top results
- **Knowledge Graph**: Entity extraction and relationship mapping
- **Conversation Memory**: Multi-turn context preservation
- **Source Citations**: Inline citations with page numbers

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** with `qwen:1.8b` model:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull qwen:1.8b
   ```

### Setup

```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Usage

```bash
# Activate environment
source rag_env/bin/activate

# Step 1: Ingest documents
python src/ingest.py

# Step 2: Start interactive chat
python src/main.py

# Or run a single query
python src/main.py --query "What are the best practices for drip irrigation?"
```

## Project Structure

```
├── documents/              # PDF documents by category
│   ├── 01_core_water_agriculture/
│   ├── 02_precision_agriculture_tech/
│   ├── 03_irrigation_systems/
│   └── ...
├── src/
│   ├── config/            # Configuration
│   │   └── config.py
│   ├── document_processing/
│   │   └── processor.py   # PDF extraction & chunking
│   ├── embeddings/
│   │   ├── generator.py   # Sentence transformer embeddings
│   │   └── vector_store.py # ChromaDB integration
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Semantic + BM25 search
│   │   ├── query_enhancer.py    # Multi-query generation
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── knowledge_graph/
│   │   └── graph_builder.py     # Entity/relationship extraction
│   ├── generation/
│   │   ├── memory_manager.py    # Conversation memory
│   │   └── rag_chain.py         # LangChain RAG pipeline
│   ├── ingest.py          # Document ingestion script
│   └── main.py            # Main application
├── data/
│   ├── vector_store/      # ChromaDB persistence
│   ├── knowledge_graph/   # Graph storage
│   └── processed_chunks/  # Processed document chunks
├── requirements.txt
└── setup.sh
```

## Configuration

Edit `src/config/config.py` to customize:

- **LLM settings**: model, temperature, context window
- **Embedding model**: all-MiniLM-L6-v2 (default) or alternatives
- **Chunk size**: 1000 characters with 200 overlap
- **Retrieval weights**: 70% semantic, 30% keyword
- **Reranking**: top-20 → top-5 with cross-encoder

## Example Queries

- "What are the best practices for drip irrigation in arid regions?"
- "How do soil moisture sensors improve water efficiency?"
- "What is evapotranspiration and how is it measured?"
- "Compare sprinkler vs drip irrigation for cotton crops"
- "What IoT technologies are used in precision agriculture?"

## Technical Details

### Retrieval Pipeline

1. **Query Enhancement**: Generate 3 query variations using LLM
2. **Hybrid Search**: Semantic (ChromaDB) + BM25 keyword search
3. **Fusion**: Reciprocal Rank Fusion (RRF) to combine results
4. **Reranking**: Cross-encoder scores top 20 → returns top 5
5. **Knowledge Graph**: Add entity context from extracted graph

### Embedding Model

- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Device**: CUDA (GPU) when available
- **Batch size**: 32 for 4GB VRAM optimization

### LLM Settings (Qwen 1.8B)

- **Temperature**: 0.1 (low for accuracy)
- **Context window**: 4096 tokens
- **Max output**: 512 tokens

## License

MIT License
