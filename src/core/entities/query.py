"""
Core Domain Entities - Query Types

Contains:
- QueryType: Classification of query intents
- ToolType: Available retrieval tools
- SubQuery: Decomposed sub-query with classification
- AgentState: Stateful memory for loop prevention
- AgentResult: Result from RAG Agent execution
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Set


class ToolType(Enum):
    """Available retrieval tools."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class QueryType(Enum):
    """Sub-query classification."""
    FACTUAL = "factual"
    RELATIONAL = "relational"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"


@dataclass
class SubQuery:
    """Decomposed sub-query with classification."""
    text: str
    query_type: QueryType
    tool: ToolType
    entities: List[str] = field(default_factory=list)
    resolved: bool = False
    context: str = ""


@dataclass
class AgentState:
    """
    Stateful memory for the agent to prevent infinite loops.
    Tracks visited entities and retrieval attempts.
    """
    visited_entities: Set[str] = field(default_factory=set)
    traversal_paths: List[List[str]] = field(default_factory=list)
    retrieval_attempts: int = 0
    max_attempts: int = 5
    context_buffer: List[str] = field(default_factory=list)
    
    def is_visited(self, entity: str) -> bool:
        return entity.lower() in self.visited_entities
    
    def mark_visited(self, entity: str):
        self.visited_entities.add(entity.lower())
    
    def can_continue(self) -> bool:
        return self.retrieval_attempts < self.max_attempts
    
    def increment_attempt(self):
        self.retrieval_attempts += 1
    
    def add_context(self, context: str):
        if context and context not in self.context_buffer:
            self.context_buffer.append(context)
    
    def get_full_context(self) -> str:
        return "\n\n".join(self.context_buffer)
    
    def reset(self):
        self.visited_entities.clear()
        self.traversal_paths.clear()
        self.retrieval_attempts = 0
        self.context_buffer.clear()


@dataclass
class AgentResult:
    """Result from the RAG Agent."""
    answer: str
    sub_queries: List[SubQuery]
    tools_used: List[str]
    hops_taken: int
    entities_visited: List[str]
    context_sources: List[str]
    confidence: float
    reasoning_trace: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sub_queries": [
                {"text": sq.text, "type": sq.query_type.value, "tool": sq.tool.value}
                for sq in self.sub_queries
            ],
            "tools_used": self.tools_used,
            "hops_taken": self.hops_taken,
            "entities_visited": self.entities_visited,
            "context_sources": self.context_sources,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace
        }
