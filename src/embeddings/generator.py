"""
Embedding Generation Module
- GPU-accelerated SentenceTransformer embeddings
- Batch processing for efficiency
- Embedding caching support
"""
import json
import pickle
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import torch

from ..config.config import EMBEDDING_CONFIG, DATA_DIR, logger


class EmbeddingGenerator:
    """Generate and manage embeddings using SentenceTransformers."""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        cache_dir: Path = None
    ):
        self.model_name = model_name or EMBEDDING_CONFIG["model_name"]
        self.device = device or EMBEDDING_CONFIG["device"]
        self.batch_size = batch_size or EMBEDDING_CONFIG["batch_size"]
        self.cache_dir = cache_dir or (DATA_DIR / "embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Embedding cache
        self._cache = {}
        self._load_cache()
    
    def _get_cache_path(self) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            use_cache: Use cached embeddings when available
        
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if use_cache:
            # Check cache for existing embeddings
            cached_indices = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self._cache:
                    cached_indices.append((i, self._cache[text_hash]))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            if cached_indices:
                logger.info(f"Using {len(cached_indices)} cached embeddings, generating {len(texts_to_embed)} new")
            
            if texts_to_embed:
                # Generate new embeddings
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=EMBEDDING_CONFIG["normalize_embeddings"]
                )
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    text_hash = self._get_text_hash(text)
                    self._cache[text_hash] = embedding
                
                # Save cache periodically
                if len(self._cache) % 100 == 0:
                    self._save_cache()
            else:
                new_embeddings = np.array([])
            
            # Combine cached and new embeddings
            embeddings = np.zeros((len(texts), self.embedding_dim))
            
            for i, embedding in cached_indices:
                embeddings[i] = embedding
            
            for idx, i in enumerate(text_indices):
                embeddings[i] = new_embeddings[idx]
            
            return embeddings
        else:
            # Generate all embeddings without cache
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=EMBEDDING_CONFIG["normalize_embeddings"]
            )
            return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
        
        Returns:
            1D numpy array of embedding
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_CONFIG["normalize_embeddings"]
        )
        return embedding
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding (1D)
            document_embeddings: Document embeddings (2D)
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        # Compute cosine similarity
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(document_embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: Path,
        metadata: dict = None
    ):
        """Save embeddings to disk."""
        np.save(output_path, embeddings)
        
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        logger.info(f"Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(input_path)
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")
        return embeddings
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache = {}
        cache_path = self._get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
        logger.info("Cleared embedding cache")
    
    def finalize(self):
        """Save cache and cleanup."""
        self._save_cache()


def main():
    """Test embedding generation."""
    generator = EmbeddingGenerator()
    
    # Test texts
    test_texts = [
        "Precision irrigation uses sensors to optimize water usage in agriculture.",
        "Drip irrigation systems deliver water directly to plant roots.",
        "Soil moisture sensors help farmers make informed irrigation decisions.",
        "Climate change is affecting water availability for crops worldwide.",
    ]
    
    print(f"Generating embeddings for {len(test_texts)} texts...")
    embeddings = generator.generate_embeddings(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query = "How do sensors help with irrigation?"
    query_embedding = generator.generate_query_embedding(query)
    
    results = generator.compute_similarity(query_embedding, embeddings, top_k=3)
    
    print(f"\nQuery: {query}")
    print("Top matches:")
    for idx, score in results:
        print(f"  [{score:.4f}] {test_texts[idx][:60]}...")
    
    generator.finalize()


if __name__ == "__main__":
    main()
