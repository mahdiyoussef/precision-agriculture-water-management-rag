"""
Core Entities Package

Central domain objects for the RAG system.
"""
from .chunk import (
    SemanticChunk,
    HierarchicalChunk,
    SentenceIndex,
    ChunkingMetrics,
)
from .query import (
    QueryType,
    ToolType,
    SubQuery,
    AgentState,
    AgentResult,
)
from .evaluation import (
    VerdictType,
    SafetyLevel,
    JudgeScore,
    ClaimVerification,
    EvaluationReport,
    AgenticMetrics,
)

__all__ = [
    # Chunk entities
    "SemanticChunk",
    "HierarchicalChunk", 
    "SentenceIndex",
    "ChunkingMetrics",
    # Query entities
    "QueryType",
    "ToolType",
    "SubQuery",
    "AgentState",
    "AgentResult",
    # Evaluation entities
    "VerdictType",
    "SafetyLevel",
    "JudgeScore",
    "ClaimVerification",
    "EvaluationReport",
    "AgenticMetrics",
]
