"""
SemanticContextAgent v2.0 - Agentic Proposition-Based Semantic Splitter
Domain: Precision Agriculture / Morocco Water Management

Features:
- Propositional decomposition (atomic thought units)
- Semantic clustering with topic vector shift detection
- Global Context Header generation (contextual anchoring)
- Intent classification (Technical, Instructional, Conceptual)
- Semantic keyword extraction
- Morocco-specific metadata tagging
- JSON array output format
"""
import json
import uuid
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import pdfplumber
from tqdm import tqdm

from ..config.config import (
    DOCUMENTS_DIR, CHUNKS_DIR, DATA_DIR,
    EMBEDDING_CONFIG, DEVICE, logger
)


# ============================================================================
# Configuration for Semantic Chunking
# ============================================================================

SEMANTIC_CHUNK_CONFIG = {
    # Token-based sizing (approx 4 chars per token)
    "target_token_range": [200, 500],
    "min_chunk_chars": 800,   # ~200 tokens
    "max_chunk_chars": 2000,  # ~500 tokens
    "target_chunk_chars": 1400,  # ~350 tokens (midpoint)
    
    # Overlap strategy: 10% semantic bridge
    "overlap_percent": 0.10,
    
    # Breakpoint detection
    "breakpoint_threshold_percentile": 95,
    "similarity_drop_threshold": 0.15,
    
    # Integrity rules
    "preserve_code_blocks": True,
    "preserve_tables": True,
    "preserve_latex": True,
}

# ============================================================================
# Advanced Feature Configurations
# ============================================================================

# 1. Contextual Chunk Headers
CONTEXTUAL_HEADER_CONFIG = {
    "enabled": True,
    "include_document_title": True,
    "include_section_hierarchy": True,
    "include_page_number": True,
    "header_template": "[Document: {doc_title}] [Section: {section}] [Page: {page}]",
}

# 2. Sliding Windows Configuration
SLIDING_WINDOW_CONFIG = {
    "enabled": True,
    "window_size": 1400,      # chars (~350 tokens)
    "stride": 1000,           # overlap = window_size - stride = 400 chars
    "min_window_content": 200,  # minimum content for valid window
}

# 3. Hierarchical Chunking Configuration
HIERARCHICAL_CONFIG = {
    "enabled": True,
    "levels": ["document", "section", "chunk", "sentence"],
    "store_parent_references": True,
    "store_children_references": True,
}

# 4. Sentence-Window Retrieval Configuration
SENTENCE_WINDOW_CONFIG = {
    "enabled": True,
    "sentences_before": 3,    # context window before
    "sentences_after": 3,     # context window after
    "store_sentence_embeddings": True,
}

# Morocco-specific metadata
MOROCCO_BASINS = [
    "Sebou", "Tensift", "Moulouya", "Oum Er-Rbia", "Souss-Massa",
    "Draa-Oued Noun", "Guir-Ziz-Rhéris", "Loukkos", "Bouregreg-Chaouia",
    "Sakia El Hamra", "Laâyoune"
]

MOROCCO_REGIONS = [
    "Tanger-Tétouan-Al Hoceïma", "Oriental", "Fès-Meknès", 
    "Rabat-Salé-Kénitra", "Béni Mellal-Khénifra", "Casablanca-Settat",
    "Marrakech-Safi", "Drâa-Tafilalet", "Souss-Massa",
    "Guelmim-Oued Noun", "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab"
]

CONTEXT_INJECTION_TEMPLATE = """[DOC_ORIGIN: {filename}]
[SUB_SECTION: {header_path}]
[LOCAL_SCOPE: Morocco / {region}]
[CONTENT]: {chunk_text}"""


# ============================================================================
# Data Structures
# ============================================================================

# Intent categories for classification
INTENT_CATEGORIES = ["Technical", "Instructional", "Conceptual"]

# Hierarchical levels
HIERARCHY_LEVELS = ["document", "section", "chunk", "sentence"]


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
    Follows SemanticContextAgent v2.0 output schema.
    Enhanced with hierarchical and contextual header support.
    """
    # Core fields (v2.0 schema)
    id: int  # Changed to integer per spec
    content: str  # Renamed from 'text' per spec
    context_summary: str  # Global Context Header
    intent_category: str = "Conceptual"  # Technical, Instructional, Conceptual
    semantic_keywords: List[str] = field(default_factory=list)  # 5 core entities
    
    # Contextual chunk header (Feature 1)
    contextual_header: str = ""  # [Document: X] [Section: Y] [Page: Z]
    content_with_header: str = ""  # Header + content combined
    
    # Hierarchical fields (Feature 3)
    level: str = "chunk"  # Hierarchy level
    parent_id: Optional[int] = None  # Parent section/document ID
    children_ids: List[int] = field(default_factory=list)  # Child sentence IDs
    
    # Sentence window data (Feature 4)
    sentences: List[str] = field(default_factory=list)  # Decomposed sentences
    sentence_ids: List[int] = field(default_factory=list)  # References to SentenceIndex
    
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
    
    # Morocco-specific
    detected_basin: Optional[str] = None
    detected_region: Optional[str] = None
    
    # Quality metrics
    boundary_score: float = 1.0
    noise_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_schema(self) -> Dict[str, Any]:
        """Output in v2.0 schema format with enhanced fields."""
        return {
            "id": self.id,
            "context_summary": self.context_summary,
            "contextual_header": self.contextual_header,
            "content": self.content,
            "content_with_header": self.content_with_header,
            "intent_category": self.intent_category,
            "semantic_keywords": self.semantic_keywords[:5],
            # Hierarchical fields
            "level": self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            # Sentence data
            "sentences": self.sentences,
        }
    
    def to_jsonl(self) -> str:
        """Output in JSONL format (legacy support)."""
        return json.dumps(self.to_schema(), ensure_ascii=False)
    
    def get_injected_text(self) -> str:
        """Get text with context injection."""
        return CONTEXT_INJECTION_TEMPLATE.format(
            filename=self.source_file,
            header_path=self.header_path or "Main Content",
            region=self.detected_region or self.detected_basin or "General",
            chunk_text=self.content
        )


# ============================================================================
# Chunking Evaluation Metrics (MoC - Mixture of Chunking Learners)
# ============================================================================

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
    cig_score: float = 0.0  # Non-redundancy measure
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


class ChunkingEvaluator:
    """
    Evaluator for semantic chunking quality using mathematical metrics.
    
    Implements:
    1. Boundary Clarity - Gradient of cosine similarity between S_i and S_{i+1}
    2. Chunk Stickiness - Intra-chunk vs inter-chunk semantic variance
    3. Information Density (CIG) - Contextual Information Gain for non-redundancy
    4. Sliding Window Change Point Detection (W=3, σ threshold)
    
    Formula Reference:
    - Boundary trigger: Similarity < μ - (k × σ), where k ∈ [1.5, 2.0]
    - Stickiness = σ²_inter / σ²_intra (higher = better chunk cohesion)
    - CIG(chunk) = H(chunk) - Σ MI(chunk, other_chunks)
    """
    
    def __init__(
        self, 
        embedding_model: SentenceTransformer = None,
        window_size: int = 3,
        k_threshold: float = 1.5
    ):
        """
        Initialize the chunking evaluator.
        
        Args:
            embedding_model: Pre-loaded sentence transformer model
            window_size: Sliding window size for change point detection (W=3 default)
            k_threshold: Standard deviation multiplier for boundary detection
        """
        if embedding_model is None:
            embedding_model = SentenceTransformer(
                EMBEDDING_CONFIG["model_name"], 
                device=DEVICE
            )
        self.embedding_model = embedding_model
        self.window_size = window_size
        self.k_threshold = k_threshold
    
    def evaluate_chunks(
        self, 
        chunks: List[Dict[str, Any]],
        return_detailed: bool = False
    ) -> ChunkingMetrics:
        """
        Evaluate chunking quality using mathematical metrics.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'sentences' keys
            return_detailed: Include per-chunk detailed scores
            
        Returns:
            ChunkingMetrics with all evaluation scores
        """
        if not chunks:
            return ChunkingMetrics()
        
        metrics = ChunkingMetrics()
        
        # Collect all texts and embeddings
        chunk_texts = [c.get("text", c.get("content", "")) for c in chunks]
        chunk_embeddings = self.embedding_model.encode(
            chunk_texts, 
            show_progress_bar=False
        )
        
        # 1. Calculate Boundary Clarity
        metrics.boundary_clarity = self._calculate_boundary_clarity(
            chunks, chunk_embeddings
        )
        
        # 2. Calculate Chunk Stickiness
        intra_var, inter_var, stickiness = self._calculate_stickiness(
            chunks, chunk_embeddings
        )
        metrics.intra_chunk_variance = intra_var
        metrics.inter_chunk_variance = inter_var
        metrics.stickiness = stickiness
        
        # 3. Calculate Contextual Information Gain (CIG)
        metrics.cig_score, metrics.entropy = self._calculate_cig(
            chunk_texts, chunk_embeddings
        )
        
        # 4. Sliding window change point detection stats
        all_similarities = self._get_all_similarities(chunks)
        if len(all_similarities) > 0:
            metrics.avg_similarity = float(np.mean(all_similarities))
            metrics.similarity_stddev = float(np.std(all_similarities))
            
            # Count boundaries detected using μ - k×σ threshold
            threshold = metrics.avg_similarity - (self.k_threshold * metrics.similarity_stddev)
            metrics.num_boundaries_detected = int(np.sum(all_similarities < threshold))
        
        # Per-chunk detailed scores
        if return_detailed:
            metrics.chunk_scores = self._calculate_per_chunk_scores(
                chunks, chunk_embeddings
            )
        
        return metrics
    
    def _calculate_boundary_clarity(
        self, 
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray
    ) -> float:
        """
        Calculate boundary clarity using gradient of cosine similarity.
        
        ∇sim(i) = sim(S_i, S_{i+1}) - sim(S_{i-1}, S_i)
        High negative gradient = strong boundary (good)
        
        Returns:
            Normalized boundary clarity score [0, 1]
        """
        if len(chunk_embeddings) < 2:
            return 1.0
        
        # Calculate inter-chunk similarities
        inter_similarities = []
        for i in range(len(chunk_embeddings) - 1):
            sim = self._cosine_similarity(
                chunk_embeddings[i], 
                chunk_embeddings[i + 1]
            )
            inter_similarities.append(sim)
        
        if not inter_similarities:
            return 1.0
        
        # Calculate gradients (∇sim)
        gradients = []
        for i in range(1, len(inter_similarities)):
            gradient = inter_similarities[i] - inter_similarities[i - 1]
            gradients.append(gradient)
        
        if not gradients:
            # Only one boundary, use raw dissimilarity
            return 1.0 - np.mean(inter_similarities)
        
        # Negative gradients indicate clear boundaries
        # Normalize: more negative gradients = higher clarity
        avg_gradient = np.mean(gradients)
        
        # Convert to [0, 1] scale where 1 = clearest boundaries
        # A negative average gradient is good (shows drops at boundaries)
        clarity = 0.5 - avg_gradient  # Center at 0.5, boost for negative gradients
        return float(np.clip(clarity, 0, 1))
    
    def _calculate_stickiness(
        self, 
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate chunk stickiness (intra vs inter variance).
        
        σ²_intra = Average variance within chunks (should be low)
        σ²_inter = Variance across chunk boundaries (should be high)
        Stickiness = σ²_inter / σ²_intra (higher = better)
        
        Returns:
            Tuple of (intra_variance, inter_variance, stickiness_ratio)
        """
        intra_variances = []
        inter_similarities = []
        
        # Calculate intra-chunk variance
        for chunk in chunks:
            sentences = chunk.get("sentences", [])
            if len(sentences) < 2:
                continue
            
            # Get sentence embeddings within chunk
            try:
                sent_embeddings = self.embedding_model.encode(
                    sentences, 
                    show_progress_bar=False
                )
                
                # Calculate pairwise similarities within chunk
                sims = []
                for i in range(len(sent_embeddings)):
                    for j in range(i + 1, len(sent_embeddings)):
                        sim = self._cosine_similarity(
                            sent_embeddings[i], 
                            sent_embeddings[j]
                        )
                        sims.append(sim)
                
                if sims:
                    intra_variances.append(np.var(sims))
            except Exception:
                continue
        
        # Calculate inter-chunk similarities
        for i in range(len(chunk_embeddings) - 1):
            sim = self._cosine_similarity(
                chunk_embeddings[i], 
                chunk_embeddings[i + 1]
            )
            inter_similarities.append(sim)
        
        # Compute final metrics
        intra_var = float(np.mean(intra_variances)) if intra_variances else 0.0
        inter_var = float(np.var(inter_similarities)) if inter_similarities else 0.0
        
        # Stickiness ratio (avoid division by zero)
        stickiness = inter_var / (intra_var + 1e-8)
        
        return intra_var, inter_var, float(stickiness)
    
    def _calculate_cig(
        self, 
        chunk_texts: List[str],
        chunk_embeddings: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Contextual Information Gain (CIG).
        
        CIG(chunk) = H(chunk) - Σ MI(chunk, other_chunks)
        
        Measures non-redundant entropy - each chunk should contain
        unique information not present in other chunks.
        
        Returns:
            Tuple of (cig_score, total_entropy)
        """
        if len(chunk_texts) < 2:
            return 1.0, 1.0
        
        # Estimate entropy via text length and vocabulary diversity
        entropies = []
        for text in chunk_texts:
            words = text.lower().split()
            if not words:
                entropies.append(0)
                continue
            
            # Vocabulary diversity as entropy proxy
            unique_ratio = len(set(words)) / len(words)
            # Length contribution (longer = more information)
            length_factor = min(1.0, len(words) / 200)
            
            entropy = unique_ratio * length_factor
            entropies.append(entropy)
        
        total_entropy = float(np.mean(entropies))
        
        # Calculate mutual information via embedding similarity
        # High similarity = high MI = redundancy (bad)
        mi_values = []
        for i in range(len(chunk_embeddings)):
            chunk_mi = []
            for j in range(len(chunk_embeddings)):
                if i != j:
                    sim = self._cosine_similarity(
                        chunk_embeddings[i], 
                        chunk_embeddings[j]
                    )
                    # Higher similarity = higher mutual information
                    chunk_mi.append(max(0, sim))
            if chunk_mi:
                mi_values.append(np.mean(chunk_mi))
        
        avg_mi = float(np.mean(mi_values)) if mi_values else 0
        
        # CIG = Entropy - Mutual Information (normalized to [0, 1])
        cig = total_entropy * (1 - avg_mi)
        
        return float(np.clip(cig, 0, 1)), total_entropy
    
    def _get_all_similarities(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Get all consecutive sentence similarities across chunks.
        Used for sliding window change point detection.
        """
        all_sentences = []
        for chunk in chunks:
            sentences = chunk.get("sentences", [])
            all_sentences.extend(sentences)
        
        if len(all_sentences) < 2:
            return np.array([])
        
        try:
            embeddings = self.embedding_model.encode(
                all_sentences, 
                show_progress_bar=False
            )
            
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            
            return np.array(similarities)
        except Exception:
            return np.array([])
    
    def _calculate_per_chunk_scores(
        self, 
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray
    ) -> List[Dict[str, float]]:
        """Calculate detailed scores for each chunk."""
        scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_score = {
                "chunk_index": i,
                "length": len(chunk.get("text", chunk.get("content", ""))),
                "sentence_count": len(chunk.get("sentences", [])),
                "boundary_score": chunk.get("boundary_score", 1.0),
            }
            
            # Similarity to neighbors
            if i > 0:
                chunk_score["similarity_to_prev"] = float(
                    self._cosine_similarity(
                        chunk_embeddings[i], 
                        chunk_embeddings[i - 1]
                    )
                )
            if i < len(chunks) - 1:
                chunk_score["similarity_to_next"] = float(
                    self._cosine_similarity(
                        chunk_embeddings[i], 
                        chunk_embeddings[i + 1]
                    )
                )
            
            scores.append(chunk_score)
        
        return scores
    
    def detect_change_points(
        self, 
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect semantic change points using sliding window (W=3).
        
        Triggers boundary where: Similarity < μ - (k × σ)
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of detected change points with metadata
        """
        if len(sentences) < self.window_size + 1:
            return []
        
        embeddings = self.embedding_model.encode(
            sentences, 
            show_progress_bar=False
        )
        
        # Calculate all consecutive similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Sliding window statistics
        change_points = []
        half_window = self.window_size // 2
        
        for i in range(half_window, len(similarities) - half_window):
            # Get window around current position
            window_start = max(0, i - half_window)
            window_end = min(len(similarities), i + half_window + 1)
            window = similarities[window_start:window_end]
            
            # Calculate local statistics
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            # Check threshold: Similarity < μ - (k × σ)
            threshold = local_mean - (self.k_threshold * local_std)
            
            if similarities[i] < threshold:
                change_points.append({
                    "position": i + 1,  # After this sentence
                    "similarity": float(similarities[i]),
                    "threshold": float(threshold),
                    "local_mean": float(local_mean),
                    "local_std": float(local_std),
                    "confidence": float((threshold - similarities[i]) / (local_std + 1e-8))
                })
        
        return change_points
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ============================================================================
# Semantic Chunker
# ============================================================================

class SemanticChunker:
    """
    SemanticContextAgent v2.0 - Agentic Proposition-Based Semantic Splitter.
    Uses embedding-based similarity to detect topic boundaries with
    intent classification and semantic keyword extraction.
    """
    
    def __init__(self, embedding_model: str = None):
        """Initialize with embedding model for semantic splitting."""
        model_name = embedding_model or EMBEDDING_CONFIG["model_name"]
        self.embedding_model = SentenceTransformer(model_name, device=DEVICE)
        
        self.config = SEMANTIC_CHUNK_CONFIG
        self.chunk_counter = 0  # For sequential integer IDs
        
        # Fallback text splitter for initial splitting
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["target_chunk_chars"],
            chunk_overlap=int(self.config["target_chunk_chars"] * self.config["overlap_percent"]),
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            length_function=len,
        )
        
        logger.info(f"SemanticContextAgent v2.0 initialized with model: {model_name}")
    
    # ============================================================================
    # Text Extraction
    # ============================================================================
    
    def extract_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text with layout analysis from PDF."""
        content = {
            "filename": pdf_path.name,
            "pages": [],
            "headers": [],
            "tables": [],
            "full_text": ""
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = {
                        "page_num": page_num,
                        "text": "",
                        "tables": [],
                        "has_table": False
                    }
                    
                    # Extract text
                    text = page.extract_text() or ""
                    page_data["text"] = text
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        page_data["has_table"] = True
                        for table in tables:
                            table_text = self._table_to_text(table)
                            page_data["tables"].append(table_text)
                            content["tables"].append({
                                "page": page_num,
                                "content": table_text
                            })
                    
                    content["pages"].append(page_data)
                    content["full_text"] += text + "\n\n"
        
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
            # Fallback to pypdf
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    content["pages"].append({
                        "page_num": page_num,
                        "text": text,
                        "tables": [],
                        "has_table": False
                    })
                    content["full_text"] += text + "\n\n"
            except Exception as e2:
                logger.error(f"Failed to extract from {pdf_path}: {e2}")
        
        # Extract headers
        content["headers"] = self._extract_headers(content["full_text"])
        
        return content
    
    def _table_to_text(self, table: List[List[Any]]) -> str:
        """Convert table to structured text."""
        rows = []
        for row in table:
            if row:
                cells = [str(cell) if cell else "" for cell in row]
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract section headers from text."""
        headers = []
        
        # Common header patterns
        patterns = [
            (r'^#+\s+(.+)$', 'markdown'),
            (r'^(\d+\.[\d\.]*)\s+(.+)$', 'numbered'),
            (r'^([A-Z][A-Z\s]{5,50})$', 'caps'),
            (r'^(Chapter|Section|Part)\s+[\dIVXLC]+[:\s]+(.+)', 'formal'),
        ]
        
        for line_num, line in enumerate(text.split('\n')):
            line = line.strip()
            for pattern, header_type in patterns:
                match = re.match(pattern, line)
                if match:
                    headers.append({
                        "text": match.group(0),
                        "type": header_type,
                        "position": line_num
                    })
                    break
        
        return headers
    
    # ============================================================================
    # Text Cleaning
    # ============================================================================
    
    def clean_text(self, text: str) -> Tuple[str, float]:
        """
        Clean text and return noise ratio.
        Returns: (cleaned_text, noise_ratio)
        """
        original_len = len(text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'Page \d+ of \d+',
            r'\d+\s*\|\s*Page',
            r'©.*?\d{4}',
            r'All rights reserved',
            r'www\.\S+',
            r'http[s]?://\S+',
            r'^\s*\d+\s*$',  # Page numbers alone
        ]
        
        noise_removed = 0
        for pattern in noise_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            noise_removed += sum(len(m) for m in matches)
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        noise_ratio = noise_removed / original_len if original_len > 0 else 0
        
        return text.strip(), noise_ratio
    
    # ============================================================================
    # Proposition Extraction
    # ============================================================================
    
    def extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic propositions (standalone factual statements).
        """
        propositions = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check if it's a factual statement (contains numbers, metrics, or specific terms)
            is_factual = bool(re.search(
                r'\d+\.?\d*\s*(%|mm|cm|m|kg|ha|L|°C|pH|kPa|bar|m³)', 
                sentence, 
                re.IGNORECASE
            ))
            
            # Check for instruction patterns
            is_instruction = bool(re.search(
                r'^(should|must|need to|require|recommend|ensure|avoid|maintain)',
                sentence,
                re.IGNORECASE
            ))
            
            if is_factual or is_instruction:
                propositions.append(sentence)
        
        return propositions
    
    # ============================================================================
    # Semantic Splitting
    # ============================================================================
    
    def calculate_similarities(self, sentences: List[str]) -> np.ndarray:
        """Calculate cosine similarities between consecutive sentences."""
        if len(sentences) < 2:
            return np.array([])
        
        embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        return np.array(similarities)
    
    def find_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """
        Find semantic breakpoints using percentile threshold.
        Breakpoints are where similarity drops significantly.
        """
        if len(similarities) == 0:
            return []
        
        # Calculate threshold (95th percentile of dissimilarity)
        dissimilarities = 1 - similarities
        threshold = np.percentile(dissimilarities, self.config["breakpoint_threshold_percentile"])
        
        # Also check for drops > 0.15
        min_drop = self.config["similarity_drop_threshold"]
        
        breakpoints = []
        for i, sim in enumerate(similarities):
            dissim = 1 - sim
            if dissim >= threshold or dissim >= min_drop:
                breakpoints.append(i + 1)  # Break AFTER this sentence
        
        return breakpoints
    
    def semantic_split(self, text: str) -> List[Dict[str, Any]]:
        """
        Perform recursive semantic splitting (Step 1 & 2 of agent pipeline).
        Step 1: Propositional split - decompose into sentences
        Step 2: Semantic clustering - group until topic vector shifts
        """
        # Step 1: Split into sentences (propositions)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 1:
            return [{"text": text, "sentences": sentences, "boundary_score": 1.0}]
        
        # Step 2: Calculate semantic similarities
        similarities = self.calculate_similarities(sentences)
        
        # Find topic shift breakpoints
        breakpoints = self.find_breakpoints(similarities)
        
        # Create chunks from breakpoints
        chunks = []
        start_idx = 0
        
        # Add end as final breakpoint
        all_breaks = sorted(set(breakpoints + [len(sentences)]))
        
        for break_idx in all_breaks:
            if break_idx <= start_idx:
                continue
            
            chunk_sentences = sentences[start_idx:break_idx]
            chunk_text = " ".join(chunk_sentences)
            
            # Check chunk size constraints (token-based)
            if len(chunk_text) < self.config["min_chunk_chars"] and chunks:
                # Merge with previous chunk
                chunks[-1]["text"] += " " + chunk_text
                chunks[-1]["sentences"].extend(chunk_sentences)
            elif len(chunk_text) > self.config["max_chunk_chars"]:
                # Split further using base splitter
                sub_chunks = self.base_splitter.split_text(chunk_text)
                for sub in sub_chunks:
                    chunks.append({
                        "text": sub,
                        "sentences": [sub],
                        "boundary_score": 0.7  # Penalty for forced split
                    })
            else:
                # Check boundary quality
                boundary_score = 1.0
                if break_idx < len(similarities):
                    # Check if we cut mid-thought
                    if similarities[break_idx - 1] > 0.8:
                        boundary_score = 0.5  # Penalty
                
                chunks.append({
                    "text": chunk_text,
                    "sentences": chunk_sentences,
                    "boundary_score": boundary_score
                })
            
            start_idx = break_idx
        
        # Apply 10% semantic bridge overlap
        chunks = self._apply_semantic_overlap(chunks)
        
        return chunks
    
    def _apply_semantic_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply 10% semantic bridge: carry over final premise of previous chunk.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlap_ratio = self.config["overlap_percent"]
        
        for i in range(1, len(chunks)):
            prev_text = chunks[i-1]["text"]
            # Get last 10% of previous chunk as bridge
            bridge_len = int(len(prev_text) * overlap_ratio)
            
            if bridge_len > 0:
                # Find sentence boundary for clean overlap
                bridge_text = prev_text[-bridge_len:]
                # Try to start at sentence boundary
                sent_start = bridge_text.find(". ")
                if sent_start > 0:
                    bridge_text = bridge_text[sent_start + 2:]
                
                if bridge_text.strip():
                    chunks[i]["text"] = bridge_text.strip() + " " + chunks[i]["text"]
        
        return chunks
    
    # ============================================================================
    # Feature 2: Sliding Windows
    # ============================================================================
    
    def sliding_window_split(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text using sliding windows with overlap.
        Window size and stride configurable via SLIDING_WINDOW_CONFIG.
        """
        config = SLIDING_WINDOW_CONFIG
        if not config["enabled"]:
            return [{"text": text, "sentences": [], "boundary_score": 1.0}]
        
        window_size = config["window_size"]
        stride = config["stride"]
        min_content = config["min_window_content"]
        
        windows = []
        start = 0
        
        while start < len(text):
            end = start + window_size
            window_text = text[start:end]
            
            # Adjust to sentence boundary if possible
            if end < len(text):
                # Try to end at sentence boundary
                last_period = window_text.rfind(". ")
                if last_period > min_content:
                    window_text = window_text[:last_period + 1]
                    end = start + last_period + 1
            
            if len(window_text.strip()) >= min_content:
                windows.append({
                    "text": window_text.strip(),
                    "sentences": re.split(r'(?<=[.!?])\s+', window_text.strip()),
                    "boundary_score": 1.0,
                    "window_start": start,
                    "window_end": end,
                })
            
            start += stride
            
            # Prevent infinite loop
            if stride <= 0:
                break
        
        logger.debug(f"Sliding window created {len(windows)} windows")
        return windows
    
    # ============================================================================
    # Feature 1: Contextual Chunk Headers
    # ============================================================================
    
    def build_contextual_header(
        self, 
        doc_title: str, 
        section: str, 
        page: int
    ) -> str:
        """
        Build contextual chunk header using template.
        Format: [Document: X] [Section: Y] [Page: Z]
        """
        config = CONTEXTUAL_HEADER_CONFIG
        if not config["enabled"]:
            return ""
        
        parts = []
        
        if config["include_document_title"] and doc_title:
            # Clean title
            clean_title = doc_title.replace(".pdf", "").replace("_", " ").replace("-", " ")
            if len(clean_title) > 50:
                clean_title = clean_title[:50] + "..."
            parts.append(f"[Document: {clean_title}]")
        
        if config["include_section_hierarchy"] and section:
            parts.append(f"[Section: {section}]")
        
        if config["include_page_number"] and page > 0:
            parts.append(f"[Page: {page}]")
        
        return " ".join(parts)
    
    def apply_contextual_header(self, chunk: SemanticChunk, doc_title: str) -> SemanticChunk:
        """Apply contextual header to a chunk."""
        header = self.build_contextual_header(
            doc_title=doc_title,
            section=chunk.header_path or "Main Content",
            page=chunk.page_numbers[0] if chunk.page_numbers else 0
        )
        chunk.contextual_header = header
        chunk.content_with_header = f"{header}\n{chunk.content}" if header else chunk.content
        return chunk
    
    # ============================================================================
    # Feature 4: Sentence Window Indexing
    # ============================================================================
    
    def build_sentence_index(
        self, 
        chunk: SemanticChunk, 
        all_sentences: List[str],
        sentence_counter: int
    ) -> Tuple[List[SentenceIndex], int]:
        """
        Build sentence index with window context for sentence-window retrieval.
        Each sentence stores references to surrounding context.
        """
        config = SENTENCE_WINDOW_CONFIG
        if not config["enabled"]:
            return [], sentence_counter
        
        sentences = re.split(r'(?<=[.!?])\s+', chunk.content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        before_count = config["sentences_before"]
        after_count = config["sentences_after"]
        
        sentence_indices = []
        sentence_ids = []
        
        for i, sentence in enumerate(sentences):
            sentence_counter += 1
            
            # Build window before
            start_idx = max(0, i - before_count)
            window_before = " ".join(sentences[start_idx:i])
            
            # Build window after
            end_idx = min(len(sentences), i + after_count + 1)
            window_after = " ".join(sentences[i+1:end_idx])
            
            # Full window context
            full_window = " ".join(sentences[start_idx:end_idx])
            
            sent_idx = SentenceIndex(
                id=sentence_counter,
                sentence=sentence,
                chunk_id=chunk.id,
                position_in_chunk=i,
                window_before=window_before,
                window_after=window_after,
                full_window=full_window,
            )
            
            sentence_indices.append(sent_idx)
            sentence_ids.append(sentence_counter)
        
        # Update chunk with sentence data
        chunk.sentences = sentences
        chunk.sentence_ids = sentence_ids
        chunk.children_ids = sentence_ids  # For hierarchical reference
        
        return sentence_indices, sentence_counter
    
    # ============================================================================
    # Morocco-Specific Detection
    # ============================================================================
    
    def detect_morocco_metadata(self, text: str) -> Dict[str, Optional[str]]:
        """Detect Morocco-specific basins and regions in text."""
        text_lower = text.lower()
        
        detected = {
            "basin": None,
            "region": None
        }
        
        # Detect basins
        for basin in MOROCCO_BASINS:
            if basin.lower() in text_lower:
                detected["basin"] = basin
                break
        
        # Detect regions
        for region in MOROCCO_REGIONS:
            if region.lower() in text_lower:
                detected["region"] = region
                break
        
        return detected
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract agricultural entities from text."""
        entities = {
            "crops": [],
            "irrigation_methods": [],
            "metrics": [],
            "sensors": [],
            "water_sources": []
        }
        
        # Crop patterns
        crop_patterns = [
            r'\b(wheat|maize|corn|tomato|olive|citrus|almond|grape|potato|onion|pepper)\b',
            r'\b(barley|rice|soybean|sunflower|sugarbeet|alfalfa)\b'
        ]
        for pattern in crop_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["crops"].extend([m.lower() for m in matches])
        
        # Irrigation methods
        irrigation_patterns = [
            r'\b(drip|sprinkler|flood|furrow|surface|subsurface|pivot|micro)\s*irrigation\b',
            r'\b(center pivot|lateral move|micro-sprinkler)\b'
        ]
        for pattern in irrigation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["irrigation_methods"].extend(matches)
        
        # Metrics (numerical values with units)
        metric_matches = re.findall(
            r'(\d+\.?\d*)\s*(mm|cm|m|kg|ha|L|°C|pH|kPa|bar|m³|%)',
            text
        )
        entities["metrics"] = [f"{m[0]} {m[1]}" for m in metric_matches]
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def classify_intent(self, text: str) -> str:
        """
        Classify chunk intent: Technical, Instructional, or Conceptual.
        Maps to query verb matching (e.g., 'How to' -> Instructional).
        """
        text_lower = text.lower()
        
        # Technical patterns: metrics, specifications, data
        technical_patterns = [
            r'\d+\.?\d*\s*(%|mm|cm|m|kg|ha|L|°C|pH|kPa|bar|m³)',
            r'\b(specification|parameter|measurement|value|rate|coefficient|ratio)\b',
            r'\b(table|figure|equation|formula|calculation)\b',
        ]
        technical_score = sum(
            1 for p in technical_patterns if re.search(p, text, re.IGNORECASE)
        )
        
        # Instructional patterns: procedures, recommendations
        instructional_patterns = [
            r'\b(should|must|need to|require|recommend|ensure|avoid|maintain)\b',
            r'\b(step|procedure|method|process|guide|instruction|how to)\b',
            r'\b(first|then|next|finally|after|before)\b',
        ]
        instructional_score = sum(
            1 for p in instructional_patterns if re.search(p, text, re.IGNORECASE)
        )
        
        # Conceptual patterns: definitions, explanations
        conceptual_patterns = [
            r'\b(is defined as|refers to|means|concept|theory|principle)\b',
            r'\b(because|therefore|thus|however|although|in order to)\b',
            r'\b(importance|significance|role|impact|effect|relationship)\b',
        ]
        conceptual_score = sum(
            1 for p in conceptual_patterns if re.search(p, text, re.IGNORECASE)
        )
        
        # Return highest scoring category
        scores = {
            "Technical": technical_score,
            "Instructional": instructional_score,
            "Conceptual": conceptual_score
        }
        
        return max(scores, key=scores.get)
    
    def extract_semantic_keywords(self, text: str, entities: Dict, topics: List[str]) -> List[str]:
        """
        Extract 5 core semantic keywords from the chunk.
        Prioritizes domain-specific entities over generic terms.
        """
        keywords = []
        
        # Priority 1: Detected entities (crops, irrigation, basins)
        for crop in entities.get("crops", [])[:2]:
            keywords.append(crop)
        
        for method in entities.get("irrigation_methods", [])[:1]:
            keywords.append(method)
        
        # Priority 2: Topics
        for topic in topics[:2]:
            if topic not in keywords:
                keywords.append(topic.replace("_", " "))
        
        # Priority 3: Common agricultural terms present in text
        ag_terms = [
            "water management", "precision agriculture", "irrigation", "soil moisture",
            "crop yield", "evapotranspiration", "groundwater", "rainfall", "drought",
            "sustainable", "efficiency", "conservation", "Morocco", "basin", "sensor"
        ]
        text_lower = text.lower()
        for term in ag_terms:
            if term.lower() in text_lower and term.lower() not in [k.lower() for k in keywords]:
                keywords.append(term)
                if len(keywords) >= 5:
                    break
        
        # Pad with generic terms if needed
        generic_terms = ["agriculture", "water", "farming", "resource", "management"]
        for term in generic_terms:
            if len(keywords) < 5 and term.lower() not in [k.lower() for k in keywords]:
                keywords.append(term)
        
        return keywords[:5]  # Ensure exactly 5
    
    # ============================================================================
    # Main Processing Pipeline
    # ============================================================================
    
    def process_document(self, pdf_path: Path) -> List[SemanticChunk]:
        """
        Process a single document into semantic chunks (v2.0 pipeline).
        Step 1: Propositional split
        Step 2: Semantic clustering  
        Step 3: Contextual anchoring (Global Context Header)
        """
        logger.info(f"Processing: {pdf_path.name}")
        
        # Extract content
        doc_content = self.extract_from_pdf(pdf_path)
        
        if not doc_content["full_text"].strip():
            logger.warning(f"No content extracted from {pdf_path.name}")
            return []
        
        # Clean text
        cleaned_text, noise_ratio = self.clean_text(doc_content["full_text"])
        
        # Semantic splitting (Step 1 & 2)
        raw_chunks = self.semantic_split(cleaned_text)
        
        # Build header path for Step 3: Contextual Anchoring
        header_path = " > ".join([h["text"][:30] for h in doc_content["headers"][:3]])
        
        # Create SemanticChunk objects
        chunks = []
        for i, raw in enumerate(raw_chunks):
            # Use sequential integer IDs (v2.0 schema)
            self.chunk_counter += 1
            chunk_id = self.chunk_counter
            
            chunk_text = raw["text"]
            
            # Extract propositions
            propositions = self.extract_propositions(chunk_text)
            
            # Detect Morocco metadata
            morocco_meta = self.detect_morocco_metadata(chunk_text)
            
            # Extract entities
            entities = self.extract_entities(chunk_text)
            
            # Extract topics
            topics = self._extract_topics(chunk_text)
            
            # Classify intent (v2.0)
            intent_category = self.classify_intent(chunk_text)
            
            # Extract semantic keywords (v2.0)
            semantic_keywords = self.extract_semantic_keywords(chunk_text, entities, topics)
            
            # Get page numbers (approximate)
            page_nums = self._get_page_numbers(chunk_text, doc_content["pages"])
            
            # Step 3: Build Global Context Header (v2.0 context_summary)
            context_summary = self._build_global_context_header(
                pdf_path.name, header_path, chunk_text, entities, morocco_meta
            )
            
            chunk = SemanticChunk(
                id=chunk_id,
                content=chunk_text,  # v2.0: 'content' instead of 'text'
                context_summary=context_summary,
                intent_category=intent_category,  # v2.0 field
                semantic_keywords=semantic_keywords,  # v2.0 field
                metadata={
                    "source": pdf_path.name,
                    "category": pdf_path.parent.name,
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "token_estimate": len(chunk_text) // 4,  # ~4 chars per token
                    "has_metrics": len(entities["metrics"]) > 0,
                    "has_table": any(p.get("has_table", False) for p in doc_content["pages"]),
                },
                source_file=pdf_path.name,
                page_numbers=page_nums,
                header_path=header_path,
                propositions=propositions,
                topics=topics,
                entities=entities,
                detected_basin=morocco_meta["basin"],
                detected_region=morocco_meta["region"],
                boundary_score=raw.get("boundary_score", 1.0),
                noise_ratio=noise_ratio
            )
            
            # Feature 1: Apply contextual header
            if CONTEXTUAL_HEADER_CONFIG["enabled"]:
                chunk = self.apply_contextual_header(chunk, pdf_path.name)
            
            chunks.append(chunk)
        
        # Feature 4: Build sentence index for all chunks
        if SENTENCE_WINDOW_CONFIG["enabled"]:
            sentence_counter = 0
            all_sentence_indices = []
            for chunk in chunks:
                sent_indices, sentence_counter = self.build_sentence_index(
                    chunk, [], sentence_counter
                )
                all_sentence_indices.extend(sent_indices)
            
            # Store sentence index in first chunk's metadata for reference
            if chunks and all_sentence_indices:
                chunks[0].metadata["total_sentences"] = len(all_sentence_indices)
        
        logger.info(f"  Created {len(chunks)} semantic chunks from {pdf_path.name}")
        return chunks
    
    def _get_page_numbers(self, chunk_text: str, pages: List[Dict]) -> List[int]:
        """Approximate which pages contain this chunk's content."""
        page_nums = []
        chunk_words = set(chunk_text.lower().split()[:20])  # First 20 words
        
        for page in pages:
            page_words = set(page["text"].lower().split())
            overlap = len(chunk_words & page_words) / len(chunk_words) if chunk_words else 0
            if overlap > 0.5:
                page_nums.append(page["page_num"])
        
        return page_nums[:2]  # Max 2 pages
    
    def _build_global_context_header(
        self, 
        filename: str,
        header_path: str,
        text: str, 
        entities: Dict, 
        morocco_meta: Dict
    ) -> str:
        """
        Build Global Context Header for Step 3: Contextual Anchoring.
        Format: 'Document: {source}; Topic: {sub-topic}'
        Prevents 'Lost in the Middle' retrieval issues.
        """
        # Document origin
        doc_name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
        # Truncate long names
        if len(doc_name) > 40:
            doc_name = doc_name[:40] + "..."
        
        # Determine sub-topic from entities and text
        sub_topics = []
        
        if morocco_meta.get("basin"):
            sub_topics.append(f"{morocco_meta['basin']} Basin")
        
        if entities.get("irrigation_methods"):
            sub_topics.append(entities["irrigation_methods"][0])
        elif "irrigation" in text.lower():
            sub_topics.append("irrigation systems")
        
        if entities.get("crops"):
            sub_topics.append(f"{entities['crops'][0]} cultivation")
        
        if not sub_topics:
            if "water" in text.lower():
                sub_topics.append("water management")
            elif "soil" in text.lower():
                sub_topics.append("soil analysis")
            else:
                sub_topics.append("agricultural practices")
        
        sub_topic = "; ".join(sub_topics[:2])
        
        return f"Document: {doc_name}; Topic: {sub_topic}"
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic tags from text."""
        topic_keywords = {
            "irrigation": ["irrigation", "watering", "drip", "sprinkler"],
            "soil": ["soil", "moisture", "texture", "ph"],
            "water_management": ["water management", "conservation", "efficiency"],
            "sensors": ["sensor", "iot", "monitoring", "smart"],
            "crops": ["crop", "plant", "yield", "harvest"],
            "climate": ["climate", "weather", "temperature", "rainfall"],
            "morocco": ["morocco", "sebou", "tensift", "rabat", "marrakech"],
        }
        
        text_lower = text.lower()
        found = []
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found.append(topic)
        
        return found
    
    def process_all_documents(
        self, 
        documents_dir: Path = DOCUMENTS_DIR,
        save_output: bool = True
    ) -> List[SemanticChunk]:
        """Process all PDFs in directory."""
        all_chunks = []
        
        pdf_files = list(documents_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in tqdm(pdf_files, desc="Semantic Chunking"):
            chunks = self.process_document(pdf_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Total semantic chunks: {len(all_chunks)}")
        
        if save_output:
            self.save_chunks(all_chunks)
        
        return all_chunks
    
    def save_chunks(
        self, 
        chunks: List[SemanticChunk],
        output_dir: Path = CHUNKS_DIR
    ):
        """
        Save chunks in v2.0 JSON array format.
        Primary output: semantic_chunks.json (v2.0 schema)
        Secondary: all_chunks.json (full details)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # v2.0 JSON Array output (primary format)
        json_path = output_dir / "semantic_chunks.json"
        schema_chunks = [c.to_schema() for c in chunks]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(schema_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {json_path} (v2.0 schema)")
        
        # JSONL output (legacy support)
        jsonl_path = output_dir / "semantic_chunks.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.to_jsonl() + '\n')
        
        # Full JSON with all metadata
        full_json_path = output_dir / "all_chunks.json"
        with open(full_json_path, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
        
        # Calculate intent distribution
        intent_counts = {}
        for c in chunks:
            intent = c.intent_category
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Save metadata summary
        metadata = {
            "version": "2.0",
            "agent": "SemanticContextAgent",
            "total_chunks": len(chunks),
            "sources": list(set(c.source_file for c in chunks)),
            "avg_chunk_size_chars": round(sum(c.metadata.get("char_count", 0) for c in chunks) / len(chunks), 1) if chunks else 0,
            "avg_token_estimate": round(sum(c.metadata.get("token_estimate", 0) for c in chunks) / len(chunks), 1) if chunks else 0,
            "intent_distribution": intent_counts,
            "basins_detected": list(set(c.detected_basin for c in chunks if c.detected_basin)),
            "regions_detected": list(set(c.detected_region for c in chunks if c.detected_region)),
            "config": self.config
        }
        
        meta_path = output_dir / "chunking_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metadata to {meta_path}")


def main():
    """Test SemanticContextAgent v2.0 chunking."""
    chunker = SemanticChunker()
    chunks = chunker.process_all_documents()
    
    print(f"\n{'='*60}")
    print("SemanticContextAgent v2.0 - Chunking Complete!")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk (v2.0 schema):")
        print(f"  ID: {sample.id}")
        print(f"  Source: {sample.source_file}")
        print(f"  Intent: {sample.intent_category}")
        print(f"  Keywords: {sample.semantic_keywords}")
        print(f"  Context: {sample.context_summary}")
        print(f"  Content preview: {sample.content[:200]}...")


if __name__ == "__main__":
    main()
