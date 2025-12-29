"""
Core Interfaces - Protocol Definitions

Abstract protocols for dependency inversion.
Allows swapping implementations without changing business logic.
"""
from typing import Protocol, List, Dict, Any, Tuple, runtime_checkable


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval implementations."""
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding generation."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        ...
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""
    
    def generate(
        self, 
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> str:
        """Generate text from prompt."""
        ...
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> str:
        """Chat completion with message history."""
        ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol for document chunking."""
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector storage."""
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to store."""
        ...
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents."""
        ...
    
    def count(self) -> int:
        """Count documents in store."""
        ...


@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """Protocol for knowledge graph operations."""
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add document to graph."""
        ...
    
    def query_entity(
        self,
        entity: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Query entity information."""
        ...
    
    def get_context_for_query(
        self,
        query: str,
        max_entities: int = 5
    ) -> str:
        """Get graph context for query."""
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for evaluation."""
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: List[str]
    ) -> Dict[str, Any]:
        """Evaluate RAG response."""
        ...
