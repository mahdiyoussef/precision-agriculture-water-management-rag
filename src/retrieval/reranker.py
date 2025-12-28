"""
Re-ranking Module
Cross-encoder for precise relevance scoring
"""
from typing import List, Dict, Any, Tuple
import numpy as np

from sentence_transformers import CrossEncoder
import torch

from ..config.config import RETRIEVAL_CONFIG, DEVICE, logger


class Reranker:
    """
    Re-rank retrieved documents using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders but slower,
    so we use them to re-score a smaller set of initial candidates.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None
    ):
        self.config = RETRIEVAL_CONFIG["reranking"]
        
        self.model_name = model_name or self.config["model"]
        self.device = device or DEVICE
        self.batch_size = batch_size or self.config["batch_size"]
        
        logger.info(f"Loading cross-encoder: {self.model_name} on {self.device}")
        
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=self.device
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on query relevance.
        
        Args:
            query: Query text
            documents: List of document dictionaries with 'text' key
            top_k: Number of top documents to return
        
        Returns:
            Reranked documents with 'rerank_score' added
        """
        if not self.config["enabled"]:
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return []
        
        top_k = top_k or self.config["top_k_final"]
        
        # Prepare query-document pairs
        pairs = [[query, doc.get('text', '')] for doc in documents]
        
        # Score all pairs
        logger.debug(f"Re-ranking {len(pairs)} documents...")
        
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            documents,
            key=lambda x: x.get('rerank_score', 0),
            reverse=True
        )
        
        # Return top k
        result = reranked[:top_k]
        
        logger.info(f"Re-ranked {len(documents)} documents, returning top {len(result)}")
        
        return result
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: Query text
            document: Document text
        
        Returns:
            Relevance score
        """
        score = self.model.predict([[query, document]])[0]
        return float(score)
    
    def batch_score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Score multiple documents against a single query.
        
        Args:
            query: Query text
            documents: List of document texts
        
        Returns:
            List of relevance scores
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]
    
    def filter_by_threshold(
        self,
        documents: List[Dict[str, Any]],
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Filter reranked documents by score threshold.
        
        Args:
            documents: Reranked documents with 'rerank_score'
            threshold: Minimum score to keep
        
        Returns:
            Filtered documents
        """
        filtered = [
            doc for doc in documents
            if doc.get('rerank_score', 0) >= threshold
        ]
        
        if len(filtered) < len(documents):
            logger.info(f"Filtered {len(documents) - len(filtered)} documents below threshold {threshold}")
        
        return filtered


def main():
    """Test reranking."""
    reranker = Reranker()
    
    query = "How do soil moisture sensors improve irrigation efficiency?"
    
    # Simulated retrieved documents
    documents = [
        {"id": "1", "text": "Soil moisture sensors measure water content in the soil, enabling precise irrigation scheduling."},
        {"id": "2", "text": "The weather in California has been unusually warm this year."},
        {"id": "3", "text": "Drip irrigation delivers water directly to plant roots, reducing evaporation losses."},
        {"id": "4", "text": "IoT sensors can monitor soil moisture in real-time and trigger irrigation automatically."},
        {"id": "5", "text": "Precision agriculture technologies help farmers optimize water usage."},
    ]
    
    print(f"Query: {query}")
    print(f"\nOriginal order:")
    for doc in documents:
        print(f"  - {doc['text'][:60]}...")
    
    reranked = reranker.rerank(query, documents, top_k=3)
    
    print(f"\nReranked (top 3):")
    for doc in reranked:
        print(f"  [{doc['rerank_score']:.4f}] {doc['text'][:60]}...")


if __name__ == "__main__":
    main()
