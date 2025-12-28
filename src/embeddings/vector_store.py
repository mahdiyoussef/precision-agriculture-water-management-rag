"""
Vector Store Module
- ChromaDB integration with persistent storage
- Batch document operations
- Similarity search with metadata filtering
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import chromadb
from chromadb.config import Settings

from ..config.config import VECTOR_STORE_CONFIG, VECTOR_STORE_DIR, logger


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        distance_metric: str = None
    ):
        self.persist_directory = persist_directory or VECTOR_STORE_CONFIG["persist_directory"]
        self.collection_name = collection_name or VECTOR_STORE_CONFIG["collection_name"]
        self.distance_metric = distance_metric or VECTOR_STORE_CONFIG["distance_metric"]
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        
        logger.info(f"Collection '{self.collection_name}' initialized with {self.collection.count()} documents")
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        batch_size: int = 100
    ):
        """
        Add documents to the vector store in batches.
        
        Args:
            ids: Unique document IDs
            embeddings: Document embeddings (2D numpy array)
            documents: Document texts
            metadatas: Document metadata dictionaries
            batch_size: Batch size for insertion
        """
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        # Convert numpy array to list
        embeddings_list = embeddings.tolist()
        
        # Clean metadata (ChromaDB only accepts str, int, float, bool)
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                elif isinstance(v, list):
                    cleaned[k] = json.dumps(v)  # Convert lists to JSON strings
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)
        
        total = len(ids)
        logger.info(f"Adding {total} documents to vector store...")
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings_list[i:end_idx]
            batch_documents = documents[i:end_idx]
            batch_metadatas = cleaned_metadatas[i:end_idx]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                logger.error(f"Error adding batch {i}-{end_idx}: {e}")
                # Try adding one by one to identify problematic documents
                for j in range(len(batch_ids)):
                    try:
                        self.collection.add(
                            ids=[batch_ids[j]],
                            embeddings=[batch_embeddings[j]],
                            documents=[batch_documents[j]],
                            metadatas=[batch_metadatas[j]]
                        )
                    except Exception as e2:
                        logger.error(f"Failed to add document {batch_ids[j]}: {e2}")
        
        logger.info(f"Vector store now contains {self.collection.count()} documents")
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        where: Dict = None,
        where_document: Dict = None,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform similarity search.
        
        Args:
            query_embedding: Query embedding (1D numpy array)
            k: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            include: What to include in results ["documents", "metadatas", "distances", "embeddings"]
        
        Returns:
            Dictionary with search results
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        query_embedding_list = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            where=where,
            where_document=where_document,
            include=include
        )
        
        return results
    
    def similarity_search_with_score(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        where: Dict = None
    ) -> List[tuple]:
        """
        Perform similarity search and return (document, score) tuples.
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            where: Metadata filter
        
        Returns:
            List of (document_dict, distance) tuples
        """
        results = self.similarity_search(
            query_embedding=query_embedding,
            k=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        scored_results = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i] if results.get('documents') else None,
                'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
            }
            # Convert distance to similarity (for cosine: similarity = 1 - distance)
            distance = results['distances'][0][i] if results.get('distances') else 0
            similarity = 1 - distance  # For cosine distance
            scored_results.append((doc, similarity))
        
        return scored_results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID."""
        results = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not results['ids']:
            return None
        
        return {
            'id': results['ids'][0],
            'text': results['documents'][0] if results.get('documents') else None,
            'metadata': results['metadatas'][0] if results.get('metadatas') else {},
            'embedding': results['embeddings'][0] if results.get('embeddings') else None,
        }
    
    def get_all_documents(
        self,
        limit: int = None,
        where: Dict = None
    ) -> List[Dict[str, Any]]:
        """Get all documents from the collection."""
        results = self.collection.get(
            limit=limit,
            where=where,
            include=["documents", "metadatas"]
        )
        
        documents = []
        for i in range(len(results['ids'])):
            documents.append({
                'id': results['ids'][i],
                'text': results['documents'][i] if results.get('documents') else None,
                'metadata': results['metadatas'][i] if results.get('metadatas') else {},
            })
        
        return documents
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def update_document(
        self,
        doc_id: str,
        embedding: np.ndarray = None,
        document: str = None,
        metadata: Dict = None
    ):
        """Update a document."""
        update_kwargs = {"ids": [doc_id]}
        
        if embedding is not None:
            update_kwargs["embeddings"] = [embedding.tolist()]
        if document is not None:
            update_kwargs["documents"] = [document]
        if metadata is not None:
            update_kwargs["metadatas"] = [metadata]
        
        self.collection.update(**update_kwargs)
        logger.info(f"Updated document {doc_id}")
    
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
    
    def reset(self):
        """Reset the collection (delete all documents)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(f"Reset collection '{self.collection_name}'")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
            "persist_directory": self.persist_directory,
        }


def main():
    """Test vector store operations."""
    from .generator import EmbeddingGenerator
    
    # Initialize
    vector_store = VectorStore()
    embedding_generator = EmbeddingGenerator()
    
    # Test data
    test_docs = [
        {"id": "test_1", "text": "Drip irrigation saves water by delivering it directly to roots."},
        {"id": "test_2", "text": "Soil moisture sensors help optimize irrigation schedules."},
        {"id": "test_3", "text": "Precision agriculture uses technology to improve farming efficiency."},
    ]
    
    # Generate embeddings
    texts = [doc["text"] for doc in test_docs]
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Add to vector store
    vector_store.add_documents(
        ids=[doc["id"] for doc in test_docs],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": "test"} for _ in test_docs]
    )
    
    print(f"Collection info: {vector_store.get_collection_info()}")
    
    # Test search
    query = "How can I save water when irrigating?"
    query_embedding = embedding_generator.generate_query_embedding(query)
    
    results = vector_store.similarity_search_with_score(query_embedding, k=3)
    
    print(f"\nQuery: {query}")
    print("Results:")
    for doc, score in results:
        print(f"  [{score:.4f}] {doc['text'][:60]}...")


if __name__ == "__main__":
    main()
