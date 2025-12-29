"""
Hybrid Retrieval System
Combines semantic search (embeddings) with keyword search (BM25)
using Reciprocal Rank Fusion (RRF)
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re

from ..config.config import RETRIEVAL_CONFIG, logger
from ..embeddings.generator import EmbeddingGenerator
from ..embeddings.vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Semantic Search - embedding-based similarity
    2. Keyword Search - BM25 algorithm
    3. Fusion - Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        documents: List[Dict[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = RETRIEVAL_CONFIG["hybrid_search"]
        
        # BM25 index
        self.bm25 = None
        self.documents = []
        self.doc_id_to_idx = {}
        
        if documents:
            self.build_bm25_index(documents)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dicts with 'id' and 'text' keys
        """
        self.documents = documents
        self.doc_id_to_idx = {doc['id']: idx for idx, doc in enumerate(documents)}
        
        # Tokenize all documents
        tokenized_docs = [self.tokenize(doc['text']) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"Built BM25 index with {len(documents)} documents")
    
    def semantic_search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (doc_id, score) tuples
        """
        top_k = top_k or self.config["top_k_semantic"]
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector store
        results = self.vector_store.similarity_search_with_score(
            query_embedding=query_embedding,
            k=top_k
        )
        
        # Return as (doc_id, score) tuples
        return [(doc['id'], score) for doc, score in results]
    
    def keyword_search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built, falling back to semantic only")
            return []
        
        top_k = top_k or self.config["top_k_keyword"]
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return as (doc_id, score) tuples
        results = [
            (self.documents[idx]['id'], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only include positive scores
        ]
        
        return results
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search and return full documents.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of document dictionaries with bm25_score
        """
        # Get (doc_id, score) tuples
        keyword_results = self.keyword_search(query, top_k)
        
        # Fetch full documents
        results = []
        for doc_id, score in keyword_results:
            if doc_id in self.doc_id_to_idx:
                doc = self.documents[self.doc_id_to_idx[doc_id]].copy()
                doc['bm25_score'] = score
                results.append(doc)
        
        return results

    
    def reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[str, float]]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion.
        
        RRF score = Σ 1 / (k + rank_i)
        
        Args:
            rankings: List of ranking lists, each containing (doc_id, score) tuples
            k: RRF constant (default 60)
        
        Returns:
            Fused ranking as (doc_id, rrf_score) tuples
        """
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def weighted_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using weighted scores.
        
        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
        
        Returns:
            Fused results as (doc_id, combined_score) tuples
        """
        semantic_weight = self.config["semantic_weight"]
        keyword_weight = self.config["keyword_weight"]
        
        combined_scores = {}
        
        # Normalize semantic scores
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            for doc_id, score in semantic_results:
                normalized = score / max_semantic if max_semantic > 0 else 0
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + semantic_weight * normalized
        
        # Normalize keyword scores
        if keyword_results:
            max_keyword = max(score for _, score in keyword_results)
            for doc_id, score in keyword_results:
                normalized = score / max_keyword if max_keyword > 0 else 0
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + keyword_weight * normalized
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_rrf: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Query text
            top_k: Number of final results
            use_rrf: Use RRF fusion (True) or weighted fusion (False)
        
        Returns:
            List of document dictionaries with scores
        """
        top_k = top_k or RETRIEVAL_CONFIG["final_top_k"]
        
        # Semantic search
        semantic_results = self.semantic_search(query)
        logger.debug(f"Semantic search returned {len(semantic_results)} results")
        
        # Keyword search
        keyword_results = self.keyword_search(query)
        logger.debug(f"Keyword search returned {len(keyword_results)} results")
        
        # Fusion
        if use_rrf:
            fused_results = self.reciprocal_rank_fusion([semantic_results, keyword_results])
        else:
            fused_results = self.weighted_fusion(semantic_results, keyword_results)
        
        # Get top k
        top_results = fused_results[:top_k]
        
        # Fetch full document information
        final_results = []
        for doc_id, score in top_results:
            # Try to get from documents list first
            if doc_id in self.doc_id_to_idx:
                doc = self.documents[self.doc_id_to_idx[doc_id]].copy()
            else:
                # Fallback to vector store
                doc = self.vector_store.get_document(doc_id)
                if doc is None:
                    continue
            
            doc['retrieval_score'] = score
            final_results.append(doc)
        
        logger.info(f"Hybrid retrieval returned {len(final_results)} results for query: {query[:50]}...")
        
        return final_results
    
    def update_index(self, new_documents: List[Dict[str, Any]]):
        """Update BM25 index with new documents."""
        if not self.documents:
            self.build_bm25_index(new_documents)
        else:
            # Add new documents
            self.documents.extend(new_documents)
            for idx, doc in enumerate(new_documents, start=len(self.doc_id_to_idx)):
                self.doc_id_to_idx[doc['id']] = idx
            
            # Rebuild BM25 index
            tokenized_docs = [self.tokenize(doc['text']) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            
            logger.info(f"Updated BM25 index, now contains {len(self.documents)} documents")


def main():
    """Test hybrid retrieval."""
    from ..document_processing.processor import DocumentProcessor
    
    # Initialize components
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Check if we have documents
    doc_count = vector_store.count()
    
    if doc_count == 0:
        print("No documents in vector store. Run ingestion first.")
        return
    
    # Get all documents for BM25
    documents = vector_store.get_all_documents()
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        documents=documents
    )
    
    # Test query
    query = "What are the best practices for drip irrigation?"
    results = retriever.retrieve(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. [Score: {doc['retrieval_score']:.4f}]")
        print(f"   Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
        print(f"   Text: {doc['text'][:200]}...")


if __name__ == "__main__":
    main()
