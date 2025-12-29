"""
Core Package

Domain layer containing:
- entities: Domain objects (chunks, queries, evaluations)
- interfaces: Protocol definitions for dependency inversion
- exceptions: Custom domain exceptions
"""
from .entities import (
    SemanticChunk,
    HierarchicalChunk,
    SentenceIndex,
    ChunkingMetrics,
    QueryType,
    ToolType,
    SubQuery,
    AgentState,
    AgentResult,
    VerdictType,
    JudgeScore,
    EvaluationReport,
    AgenticMetrics,
)
from .interfaces import (
    RetrieverProtocol,
    EmbedderProtocol,
    LLMProtocol,
    ChunkerProtocol,
    VectorStoreProtocol,
    KnowledgeGraphProtocol,
    EvaluatorProtocol,
)
from .exceptions import (
    RAGException,
    DocumentProcessingError,
    ChunkingError,
    EmbeddingError,
    RetrievalError,
    GenerationError,
    GraphQueryError,
    EvaluationError,
    ConfigurationError,
    LLMConnectionError,
)

__all__ = [
    # Entities
    "SemanticChunk",
    "HierarchicalChunk",
    "SentenceIndex", 
    "ChunkingMetrics",
    "QueryType",
    "ToolType",
    "SubQuery",
    "AgentState",
    "AgentResult",
    "VerdictType",
    "JudgeScore",
    "EvaluationReport",
    "AgenticMetrics",
    # Interfaces
    "RetrieverProtocol",
    "EmbedderProtocol",
    "LLMProtocol",
    "ChunkerProtocol",
    "VectorStoreProtocol",
    "KnowledgeGraphProtocol",
    "EvaluatorProtocol",
    # Exceptions
    "RAGException",
    "DocumentProcessingError",
    "ChunkingError",
    "EmbeddingError",
    "RetrievalError",
    "GenerationError",
    "GraphQueryError",
    "EvaluationError",
    "ConfigurationError",
    "LLMConnectionError",
]
