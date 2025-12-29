"""
Core Domain Entities - Chunk Types

Contains:
- SemanticChunk: Primary chunk with rich metadata
- HierarchicalChunk: Multi-level chunk with parent-child relationships
- SentenceIndex: Individual sentence with window context
- ChunkingMetrics: Evaluation metrics for chunking quality
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class HierarchicalChunk:
    """
    Multi-level chunk with parent-child relationships for hierarchical chunking.
    Enables navigation through document structure.
    """
    id: int
    level: str  # "document", "section", "chunk", "sentence"
    content: str
    context_header: str = ""  # Contextual chunk header
    
    # Hierarchy references
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    
    # Level-specific metadata
    level_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source
    source_file: str = ""
    page_numbers: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceIndex:
    """
    Individual sentence with window context for sentence-window retrieval.
    Stores sentence position for expanding to surrounding context.
    """
    id: int
    sentence: str
    chunk_id: int  # Parent chunk reference
    position_in_chunk: int  # Sentence position within chunk
    
    # Pre-computed window context
    window_before: str = ""  # N sentences before
    window_after: str = ""   # N sentences after
    full_window: str = ""    # Complete context window
    
    # Embedding (optional, computed on demand)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticChunk:
    """
    Represents a semantic proposition chunk with rich metadata.
    Primary chunk type for the RAG system.
    """
    # Core fields
    id: int
    content: str
    context_summary: str  # Global Context Header
    intent_category: str = "Conceptual"  # Technical, Instructional, Conceptual
    semantic_keywords: List[str] = field(default_factory=list)
    
    # Contextual chunk header
    contextual_header: str = ""
    content_with_header: str = ""
    
    # Hierarchical fields
    level: str = "chunk"
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    
    # Sentence window data
    sentences: List[str] = field(default_factory=list)
    sentence_ids: List[int] = field(default_factory=list)
    
    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    source_file: str = ""
    page_numbers: List[int] = field(default_factory=list)
    header_path: str = ""
    
    # Content analysis
    propositions: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    # Domain-specific (Morocco)
    detected_basin: Optional[str] = None
    detected_region: Optional[str] = None
    
    # Quality metrics
    boundary_score: float = 1.0
    noise_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_schema(self) -> Dict[str, Any]:
        """Output in v2.0 schema format."""
        return {
            "id": self.id,
            "context_summary": self.context_summary,
            "contextual_header": self.contextual_header,
            "content": self.content,
            "content_with_header": self.content_with_header,
            "intent_category": self.intent_category,
            "semantic_keywords": self.semantic_keywords[:5],
            "level": self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "sentences": self.sentences,
        }
    
    def get_injected_text(self, template: str = None) -> str:
        """Get text with context injection."""
        if template is None:
            template = "[DOC_ORIGIN: {filename}]\n[SUB_SECTION: {header_path}]\n[LOCAL_SCOPE: Morocco / {region}]\n[CONTENT]: {chunk_text}"
        return template.format(
            filename=self.source_file,
            header_path=self.header_path or "Main Content",
            region=self.detected_region or self.detected_basin or "General",
            chunk_text=self.content
        )


@dataclass
class ChunkingMetrics:
    """
    Mathematical evaluation metrics for semantic chunking quality.
    Implements Semantic Similarity Change Point Detection methodology.
    """
    # Boundary Clarity: gradient of cosine similarity (∇sim)
    boundary_clarity: float = 0.0
    
    # Chunk Stickiness: σ²_inter / σ²_intra (higher is better)
    stickiness: float = 0.0
    intra_chunk_variance: float = 0.0
    inter_chunk_variance: float = 0.0
    
    # Information Density: Contextual Information Gain
    cig_score: float = 0.0
    entropy: float = 0.0
    
    # Change Point Detection stats
    num_boundaries_detected: int = 0
    avg_similarity: float = 0.0
    similarity_stddev: float = 0.0
    
    # Per-chunk scores
    chunk_scores: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Boundary Clarity: {self.boundary_clarity:.3f} | "
            f"Stickiness: {self.stickiness:.3f} | "
            f"CIG: {self.cig_score:.3f}"
        )
