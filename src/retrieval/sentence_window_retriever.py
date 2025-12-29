"""
Sentence-Window Retrieval System
Implements sentence-level retrieval with context window expansion.

Feature 4: Retrieves at sentence level, then expands to surrounding sentences.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from sentence_transformers import SentenceTransformer

from ..config.config import RETRIEVAL_CONFIG, EMBEDDING_CONFIG, DEVICE, logger
from ..embeddings.generator import EmbeddingGenerator
from ..embeddings.vector_store import VectorStore


# Configuration
SENTENCE_WINDOW_CONFIG = {
    "sentences_before": 3,
    "sentences_after": 3,
    "min_sentence_length": 10,
    "similarity_threshold": 0.5,
}


@dataclass
class SentenceWithContext:
    """A sentence with its surrounding context window."""
    id: int
    sentence: str
    chunk_id: int
    source_file: str
    position_in_chunk: int
    
    # Context windows
    window_before: str  # N sentences before
    window_after: str   # N sentences after
    full_window: str    # Complete context
    
    # Retrieval metadata
    similarity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentenceWindowRetriever:
    """
    Sentence-Window Retrieval System.
    
    Process:
    1. Index individual sentences with embeddings
    2. Retrieve matching sentences based on query
    3. Expand each sentence to N surrounding sentences
    4. Return context-enriched results
    """
    
    def __init__(
        self, 
        embedding_model: str = None,
        sentences_before: int = 3,
        sentences_after: int = 3
    ):
        """
        Initialize sentence-window retriever.
        
        Args:
            embedding_model: Model for generating embeddings
            sentences_before: Number of sentences to include before match
            sentences_after: Number of sentences to include after match
        """
        model_name = embedding_model or EMBEDDING_CONFIG["model_name"]
        self.embedding_model = SentenceTransformer(model_name, device=DEVICE)
        
        self.sentences_before = sentences_before
        self.sentences_after = sentences_after
        self.min_sentence_length = SENTENCE_WINDOW_CONFIG["min_sentence_length"]
        
        # Sentence index storage
        self.sentence_index: List[Dict[str, Any]] = []
        self.sentence_embeddings: Optional[np.ndarray] = None
        
        # Chunk reference for context expansion
        self.chunks_by_id: Dict[int, Dict[str, Any]] = {}
        
        logger.info(f"SentenceWindowRetriever initialized with window: ±{sentences_before}/{sentences_after}")
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Build sentence index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'sentences'
            
        Returns:
            Number of sentences indexed
        """
        self.sentence_index = []
        self.chunks_by_id = {}
        
        sentence_id = 0
        
        for chunk in chunks:
            chunk_id = chunk.get("id", 0)
            content = chunk.get("content", "")
            source = chunk.get("source_file", "unknown")
            
            # Store chunk for context lookups
            self.chunks_by_id[chunk_id] = chunk
            
            # Get sentences (from chunk if available, otherwise extract)
            sentences = chunk.get("sentences", [])
            if not sentences:
                sentences = self._extract_sentences(content)
            
            # Index each sentence with context
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < self.min_sentence_length:
                    continue
                
                sentence_id += 1
                
                # Build context windows
                start_idx = max(0, i - self.sentences_before)
                end_idx = min(len(sentences), i + self.sentences_after + 1)
                
                window_before = " ".join(sentences[start_idx:i])
                window_after = " ".join(sentences[i+1:end_idx])
                full_window = " ".join(sentences[start_idx:end_idx])
                
                self.sentence_index.append({
                    "id": sentence_id,
                    "sentence": sentence.strip(),
                    "chunk_id": chunk_id,
                    "source_file": source,
                    "position_in_chunk": i,
                    "window_before": window_before,
                    "window_after": window_after,
                    "full_window": full_window,
                })
        
        # Generate embeddings for all sentences
        if self.sentence_index:
            sentences_text = [s["sentence"] for s in self.sentence_index]
            self.sentence_embeddings = self.embedding_model.encode(
                sentences_text,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Generated embeddings for {len(sentences_text)} sentences")
        
        logger.info(f"Built sentence index with {len(self.sentence_index)} sentences from {len(chunks)} chunks")
        return len(self.sentence_index)
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        return_window: bool = True
    ) -> List[SentenceWithContext]:
        """
        Retrieve sentences matching query with context windows.
        
        Args:
            query: Search query
            top_k: Number of sentences to retrieve
            return_window: If True, include surrounding context
            
        Returns:
            List of SentenceWithContext objects
        """
        if not self.sentence_index or self.sentence_embeddings is None:
            logger.warning("Sentence index not built. Call build_index first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate similarities
        similarities = np.dot(self.sentence_embeddings, query_embedding)
        similarities = similarities / (
            np.linalg.norm(self.sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            sent_data = self.sentence_index[idx]
            score = float(similarities[idx])
            
            if score < SENTENCE_WINDOW_CONFIG["similarity_threshold"]:
                continue
            
            result = SentenceWithContext(
                id=sent_data["id"],
                sentence=sent_data["sentence"],
                chunk_id=sent_data["chunk_id"],
                source_file=sent_data["source_file"],
                position_in_chunk=sent_data["position_in_chunk"],
                window_before=sent_data["window_before"] if return_window else "",
                window_after=sent_data["window_after"] if return_window else "",
                full_window=sent_data["full_window"] if return_window else sent_data["sentence"],
                similarity_score=score,
            )
            results.append(result)
        
        logger.debug(f"Retrieved {len(results)} sentences for query: {query[:50]}...")
        return results
    
    def retrieve_with_expansion(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and expand to full context windows.
        
        Returns results with expanded context suitable for LLM context.
        """
        sentences = self.retrieve(query, top_k=top_k, return_window=True)
        
        expanded_results = []
        for sent in sentences:
            expanded_results.append({
                "sentence": sent.sentence,
                "context": sent.full_window,
                "source": sent.source_file,
                "chunk_id": sent.chunk_id,
                "score": sent.similarity_score,
            })
        
        return expanded_results
    
    def save_index(self, output_path: Path):
        """Save sentence index to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.sentence_index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved sentence index to {output_path}")
    
    def load_index(self, index_path: Path):
        """Load sentence index from JSON and rebuild embeddings."""
        with open(index_path, 'r', encoding='utf-8') as f:
            self.sentence_index = json.load(f)
        
        if self.sentence_index:
            sentences_text = [s["sentence"] for s in self.sentence_index]
            self.sentence_embeddings = self.embedding_model.encode(
                sentences_text,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        logger.info(f"Loaded sentence index with {len(self.sentence_index)} sentences")


def main():
    """Test sentence-window retrieval."""
    from ..document_processing.semantic_chunker import SemanticChunker
    
    # Process a sample document
    chunker = SemanticChunker()
    sample_pdf = Path("documents/Water Management/water scarcity in morocco.pdf")
    
    if not sample_pdf.exists():
        print("Sample document not found")
        return
    
    # Get chunks
    chunks = chunker.process_document(sample_pdf)
    chunk_dicts = [c.to_dict() for c in chunks]
    
    # Build sentence index
    retriever = SentenceWindowRetriever()
    retriever.build_index(chunk_dicts)
    
    # Test query
    query = "What are the causes of water scarcity in Morocco?"
    results = retriever.retrieve_with_expansion(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} sentence-window results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result['score']:.4f}]")
        print(f"   Source: {result['source']}")
        print(f"   Sentence: {result['sentence'][:100]}...")
        print(f"   Context: {result['context'][:200]}...")


if __name__ == "__main__":
    main()
