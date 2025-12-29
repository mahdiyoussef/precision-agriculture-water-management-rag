"""
Core Domain Exceptions

Custom exceptions for the RAG system.
"""


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class DocumentProcessingError(RAGException):
    """Error during document processing."""
    pass


class ChunkingError(RAGException):
    """Error during text chunking."""
    pass


class EmbeddingError(RAGException):
    """Error during embedding generation."""
    pass


class RetrievalError(RAGException):
    """Error during document retrieval."""
    pass


class GenerationError(RAGException):
    """Error during response generation."""
    pass


class GraphQueryError(RAGException):
    """Error during knowledge graph query."""
    pass


class EvaluationError(RAGException):
    """Error during evaluation."""
    pass


class ConfigurationError(RAGException):
    """Invalid configuration."""
    pass


class LLMConnectionError(RAGException):
    """Cannot connect to LLM service."""
    pass
