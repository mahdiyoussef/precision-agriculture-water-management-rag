"""
Main RAG System Application
- Complete RAG pipeline
- Interactive query interface
- Document ingestion support
"""
import sys
import json
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import logger
from src.document_processing.processor import DocumentProcessor
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_enhancer import QueryEnhancer
from src.retrieval.reranker import Reranker
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.generation.memory_manager import MemoryManager
from src.generation.rag_chain import RAGChain


class PrecisionAgricultureRAG:
    """
    Complete RAG system for Precision Agriculture Water Management.
    
    Combines:
    - Hybrid retrieval (semantic + keyword)
    - Query enhancement (multi-query, expansion)
    - Cross-encoder reranking
    - Knowledge graph context
    - Conversation memory
    - LLM generation with citations
    """
    
    def __init__(self, load_existing: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            load_existing: Load existing vector store and knowledge graph
        """
        logger.info("Initializing Precision Agriculture RAG System...")
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.query_enhancer = QueryEnhancer()
        self.reranker = Reranker()
        self.memory = MemoryManager()
        
        # Hybrid retriever (initialized after loading documents)
        self.hybrid_retriever: Optional[HybridRetriever] = None
        
        # RAG chain
        self.rag_chain: Optional[RAGChain] = None
        
        if load_existing:
            self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing vector store and knowledge graph."""
        # Check if vector store has documents
        doc_count = self.vector_store.count()
        
        if doc_count == 0:
            logger.warning("Vector store is empty. Run ingestion first.")
            return
        
        logger.info(f"Loaded vector store with {doc_count} documents")
        
        # Load knowledge graph
        self.knowledge_graph.load()
        
        # Get all documents for hybrid retriever
        documents = self.vector_store.get_all_documents()
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            documents=documents
        )
        
        # Initialize RAG chain
        self.rag_chain = RAGChain(
            retriever=self.hybrid_retriever,
            memory=self.memory
        )
        
        # Attach knowledge graph to retriever for context
        self.hybrid_retriever.knowledge_graph = self.knowledge_graph
        
        logger.info("RAG system initialized and ready!")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_query_enhancement: bool = True,
        use_kg_context: bool = True
    ) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of final documents
            use_reranking: Apply cross-encoder reranking
            use_query_enhancement: Use multi-query and expansion
            use_kg_context: Include knowledge graph context
        
        Returns:
            Response dictionary with answer, sources, confidence
        """
        if self.hybrid_retriever is None:
            return {
                "answer": "System not initialized. Please run document ingestion first.",
                "sources": [],
                "confidence": 0.0
            }
        
        logger.info(f"Processing query: {question[:50]}...")
        
        # Query enhancement
        all_queries = [question]
        if use_query_enhancement:
            all_queries = self.query_enhancer.get_all_queries(question)
            logger.info(f"Generated {len(all_queries)} query variations")
        
        # Retrieve documents using all query variations
        all_docs = {}
        for q in all_queries:
            docs = self.hybrid_retriever.retrieve(q, top_k=top_k * 2)
            for doc in docs:
                doc_id = doc.get("id", doc.get("chunk_id"))
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
                else:
                    # Keep higher score
                    if doc.get("retrieval_score", 0) > all_docs[doc_id].get("retrieval_score", 0):
                        all_docs[doc_id] = doc
        
        # Convert to list and sort by score
        documents = sorted(
            all_docs.values(),
            key=lambda x: x.get("retrieval_score", 0),
            reverse=True
        )[:top_k * 2]  # Keep more for reranking
        
        # Reranking
        if use_reranking and documents:
            documents = self.reranker.rerank(question, documents, top_k=top_k)
            logger.info(f"Reranked to {len(documents)} documents")
        else:
            documents = documents[:top_k]
        
        # Get knowledge graph context
        kg_context = ""
        if use_kg_context:
            kg_context = self.knowledge_graph.get_context_for_query(question)
        
        # Generate response
        response = self.rag_chain.generate_response(
            question=question,
            documents=documents,
            knowledge_graph_context=kg_context,
            use_memory=True
        )
        
        return response
    
    def chat(self):
        """Interactive chat loop."""
        print("\n" + "="*60)
        print("PRECISION AGRICULTURE WATER MANAGEMENT RAG SYSTEM")
        print("="*60)
        print("Ask questions about water management, irrigation, sensors, etc.")
        print("Commands: /clear (clear memory), /quit (exit)")
        print("="*60 + "\n")
        
        if self.hybrid_retriever is None:
            print("WARNING: System not fully initialized. Please run ingestion first:")
            print("   python src/ingest.py")
            print()
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() == "/quit":
                    print("Goodbye!")
                    break
                elif question.lower() == "/clear":
                    self.memory.clear()
                    print("Memory cleared.\n")
                    continue
                elif question.lower() == "/help":
                    print("Commands: /clear, /quit, /help")
                    continue
                
                # Query the system
                response = self.query(question)
                
                print(f"\nAssistant: {response['answer']}")
                
                if response.get('sources'):
                    print(f"\nSources: {', '.join(set(response['sources'][:3]))}")
                
                print(f"Confidence: {response['confidence']:.0%}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}\n")
    
    def get_system_info(self) -> dict:
        """Get system information and statistics."""
        info = {
            "vector_store_count": self.vector_store.count(),
            "knowledge_graph_stats": self.knowledge_graph.get_statistics() if self.knowledge_graph.graph.number_of_nodes() > 0 else {},
            "memory_size": len(self.memory.get_messages()) if self.memory else 0,
        }
        return info


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Precision Agriculture RAG System")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single query"
    )
    args = parser.parse_args()
    
    # Initialize system
    rag = PrecisionAgricultureRAG()
    
    if args.info:
        info = rag.get_system_info()
        print(json.dumps(info, indent=2, default=str))
    elif args.query:
        response = rag.query(args.query)
        print(f"Answer: {response['answer']}")
        print(f"Sources: {response['sources']}")
        print(f"Confidence: {response['confidence']}")
    else:
        rag.chat()


if __name__ == "__main__":
    main()
