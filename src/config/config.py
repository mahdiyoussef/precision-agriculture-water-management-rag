"""
Configuration for Advanced RAG System
Optimized for: i5-10300H, 16GB RAM, GTX 1650 (4GB VRAM)
"""
from pathlib import Path
import torch
import logging

# ========== PATHS ==========
BASE_DIR = Path(__file__).parent.parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
KNOWLEDGE_GRAPH_DIR = DATA_DIR / "knowledge_graph"
CHUNKS_DIR = DATA_DIR / "processed_chunks"
METADATA_DIR = DATA_DIR / "metadata"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [VECTOR_STORE_DIR, KNOWLEDGE_GRAPH_DIR, CHUNKS_DIR, METADATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========== DEVICE CONFIGURATION ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LLM CONFIGURATION ==========
LLM_CONFIG = {
    "model": "qwen:1.8b",
    "base_url": "http://localhost:11434",
    "temperature": 0.1,  # Low for accuracy
    "top_p": 0.9,
    "top_k": 40,
    "num_ctx": 4096,  # Context window
    "repeat_penalty": 1.1,
    "num_predict": 512,  # Max output tokens
}

# ========== EMBEDDING CONFIGURATION ==========
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # Fast, 384 dimensions
    "device": DEVICE,
    "batch_size": 32,  # Optimize for 4GB VRAM
    "normalize_embeddings": True,
    "show_progress_bar": True,
}

# ========== DOCUMENT PROCESSING ==========
CHUNK_CONFIG = {
    "chunk_size": 1000,  # characters
    "chunk_overlap": 200,  # characters
    "separators": ["\n\n", "\n", ". ", " ", ""],
    "length_function": len,
    "add_start_index": True,
}

# ========== VECTOR STORE CONFIGURATION ==========
VECTOR_STORE_CONFIG = {
    "type": "chroma",
    "collection_name": "water_management_docs",
    "distance_metric": "cosine",
    "persist_directory": str(VECTOR_STORE_DIR),
    "embedding_dimension": 384,  # Match all-MiniLM-L6-v2
}

# ========== RETRIEVAL CONFIGURATION ==========
RETRIEVAL_CONFIG = {
    # Hybrid Search
    "hybrid_search": {
        "enabled": True,
        "semantic_weight": 0.7,  # 70% semantic, 30% keyword
        "keyword_weight": 0.3,
        "top_k_semantic": 20,
        "top_k_keyword": 10,
    },
    
    # Multi-Query
    "multi_query": {
        "enabled": True,
        "num_queries": 3,  # Generate 3 variations
        "aggregation_method": "reciprocal_rank_fusion",
    },
    
    # Re-ranking
    "reranking": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k_initial": 20,
        "top_k_final": 5,
        "batch_size": 8,  # Memory optimization
    },
    
    # Query Enhancement
    "query_enhancement": {
        "expansion": True,
        "rewriting": True,
        "max_expansions": 3,
    },
    
    # Final retrieval
    "final_top_k": 5,
}

# ========== KNOWLEDGE GRAPH CONFIGURATION ==========
KNOWLEDGE_GRAPH_CONFIG = {
    "enabled": True,
    "entity_types": [
        "crop", "irrigation_method", "sensor", "technology",
        "region", "water_source", "metric", "organization"
    ],
    "relation_types": [
        "requires", "improves", "measures", "located_in",
        "uses", "produces", "affects", "compatible_with"
    ],
    "extraction_model": "en_core_web_sm",
    "min_entity_frequency": 2,
    "graph_database": None,  # Use NetworkX in-memory
}

# ========== CONVERSATION MEMORY ==========
MEMORY_CONFIG = {
    "enabled": True,
    "type": "conversation_buffer_window",
    "k": 5,  # Remember last 5 exchanges
    "return_messages": True,
    "memory_key": "chat_history",
}

# ========== CITATION TRACKING ==========
CITATION_CONFIG = {
    "enabled": True,
    "format": "inline",
    "include_page_numbers": True,
    "include_document_title": True,
    "max_citations_per_response": 5,
}

# ========== GENERATION CONFIGURATION ==========
SYSTEM_PROMPT = """You are an expert agricultural water management advisor specializing in precision agriculture.

Your expertise includes:
- Irrigation systems and scheduling
- Soil moisture management
- Water conservation techniques
- IoT sensors and monitoring
- Crop-specific water requirements
- Sustainable water practices

Guidelines:
1. Base answers ONLY on provided context from technical documents
2. Cite sources with [Source: document_name, page X]
3. Provide quantitative metrics when available
4. Consider regional and crop-specific variations
5. Flag uncertainties or conflicting information
6. Prioritize practical, actionable recommendations
7. Use technical terminology appropriately

If information is not in the context, clearly state this limitation."""

GENERATION_CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "response_format": {
        "include_sources": True,
        "include_confidence": True,
        "include_related_topics": True,
        "max_length": 1000,
    },
}

# ========== LOGGING ==========
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": {
            "enabled": True,
            "filename": str(LOGS_DIR / "rag_system.log"),
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5,
        },
        "console": {
            "enabled": True,
        }
    }
}

# ========== PERFORMANCE OPTIMIZATION ==========
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_embeddings": True,
    "cache_llm_responses": True,
    "cache_ttl": 3600,  # 1 hour
    "max_workers": 4,  # Parallel processing
    "gpu_memory_fraction": 0.8,  # Use 80% of 4GB VRAM
}

# ========== DOMAIN-SPECIFIC SYNONYMS ==========
DOMAIN_SYNONYMS = {
    "irrigation": ["watering", "water application", "water delivery"],
    "drip": ["trickle", "micro-irrigation", "drip irrigation"],
    "sensor": ["monitor", "detector", "measurement device", "probe"],
    "efficiency": ["productivity", "optimization", "performance"],
    "soil moisture": ["soil water content", "soil wetness", "volumetric water content"],
    "evapotranspiration": ["ET", "ET0", "water loss", "crop water use"],
    "precision agriculture": ["smart farming", "precision farming", "digital agriculture"],
    "fertigation": ["fertilizer irrigation", "nutrient delivery"],
    "deficit irrigation": ["regulated deficit irrigation", "RDI", "controlled water stress"],
    "crop water requirement": ["CWR", "water demand", "irrigation requirement"],
}

def setup_logging():
    """Configure logging based on settings."""
    handlers = []
    
    if LOGGING_CONFIG["handlers"]["console"]["enabled"]:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGGING_CONFIG["level"])
        console_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["format"]))
        handlers.append(console_handler)
    
    if LOGGING_CONFIG["handlers"]["file"]["enabled"]:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            LOGGING_CONFIG["handlers"]["file"]["filename"],
            maxBytes=LOGGING_CONFIG["handlers"]["file"]["max_bytes"],
            backupCount=LOGGING_CONFIG["handlers"]["file"]["backup_count"]
        )
        file_handler.setLevel(LOGGING_CONFIG["level"])
        file_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["format"]))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        handlers=handlers
    )
    
    return logging.getLogger("rag_system")

# Initialize logger
logger = setup_logging()
