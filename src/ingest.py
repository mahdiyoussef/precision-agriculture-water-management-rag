"""
Document Ingestion Pipeline
- Process all PDFs
- Generate embeddings
- Store in vector database
- Build knowledge graph
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import DOCUMENTS_DIR, logger
from src.document_processing.processor import DocumentProcessor
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder


def ingest_documents(reset_store: bool = False):
    """
    Run the complete document ingestion pipeline.
    
    Args:
        reset_store: If True, reset vector store before ingestion
    """
    start_time = time.time()
    
    print("="*60)
    print("DOCUMENT INGESTION PIPELINE")
    print("="*60)
    
    # Step 1: Process documents
    print("\n[1/4] Processing PDF documents...")
    processor = DocumentProcessor()
    chunks = processor.process_all_documents(save_chunks=True)
    
    if not chunks:
        print("ERROR: No chunks created. Check if PDFs exist in documents folder.")
        return
    
    print(f"  [OK] Created {len(chunks)} chunks from {len(set(c.source for c in chunks))} documents")
    
    # Step 2: Generate embeddings
    print("\n[2/4] Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_generator.generate_embeddings(texts, show_progress=True)
    embedding_generator.finalize()
    
    print(f"  [OK] Generated embeddings with shape {embeddings.shape}")
    
    # Step 3: Store in vector database
    print("\n[3/4] Storing in ChromaDB...")
    vector_store = VectorStore()
    
    if reset_store:
        print("  Resetting vector store...")
        vector_store.reset()
    
    # Prepare data
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [chunk.to_dict() for chunk in chunks]
    
    # Remove 'text' from metadata to avoid duplication
    for meta in metadatas:
        meta.pop('text', None)
    
    vector_store.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"  [OK] Vector store contains {vector_store.count()} documents")
    
    # Step 4: Build knowledge graph
    print("\n[4/4] Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder()
    
    chunk_dicts = [
        {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.to_dict()}
        for c in chunks
    ]
    kg_builder.build_from_chunks(chunk_dicts)
    kg_builder.save()
    
    stats = kg_builder.get_statistics()
    print(f"  [OK] Knowledge graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("INGESTION COMPLETE!")
    print("="*60)
    print(f"  Documents processed: {len(set(c.source for c in chunks))}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Vector store size: {vector_store.count()}")
    print(f"  Knowledge graph nodes: {stats['num_nodes']}")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print("="*60)


def main():
    """Entry point for ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before ingestion"
    )
    args = parser.parse_args()
    
    ingest_documents(reset_store=args.reset)


if __name__ == "__main__":
    main()
