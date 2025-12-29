#!/usr/bin/env python3
"""
Re-ingestion Script for Semantic Chunking
Clears existing data and re-processes documents with advanced semantic chunking.

Usage:
    python src/reingest.py [--confirm]
"""
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import (
    VECTOR_STORE_DIR, KNOWLEDGE_GRAPH_DIR, CHUNKS_DIR, 
    METADATA_DIR, DATA_DIR, DOCUMENTS_DIR, logger
)
from src.document_processing.semantic_chunker import SemanticChunker
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder


def clear_existing_data(confirm: bool = False) -> bool:
    """
    Clear all existing processed data.
    Returns True if cleared, False if cancelled.
    """
    dirs_to_clear = [
        VECTOR_STORE_DIR,
        KNOWLEDGE_GRAPH_DIR, 
        CHUNKS_DIR,
        METADATA_DIR
    ]
    
    print("\n" + "="*60)
    print("DATA CLEARING")
    print("="*60)
    print("The following directories will be cleared:")
    
    total_size = 0
    for dir_path in dirs_to_clear:
        if dir_path.exists():
            size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            total_size += size
            print(f"  - {dir_path.relative_to(DATA_DIR.parent)}: {size/1024/1024:.2f} MB")
        else:
            print(f"  - {dir_path.relative_to(DATA_DIR.parent)}: (empty)")
    
    print(f"\nTotal size to clear: {total_size/1024/1024:.2f} MB")
    
    if not confirm:
        response = input("\nProceed with clearing? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Aborted.")
            return False
    
    # Clear directories
    for dir_path in dirs_to_clear:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared: {dir_path}")
    
    print("[OK] All data cleared successfully.")
    return True


def run_semantic_ingestion():
    """Run the full semantic ingestion pipeline."""
    print("\n" + "="*60)
    print("SEMANTIC DOCUMENT INGESTION")
    print("="*60)
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Chunking strategy: Recursive Semantic Splitting (RSC)")
    print(f"Target chunk size: 600 characters")
    print(f"Overlap: 15% semantic sliding window")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    
    # Step 1: Semantic Chunking
    print("Step 1/4: Semantic Document Chunking...")
    chunker = SemanticChunker()
    chunks = chunker.process_all_documents(DOCUMENTS_DIR)
    
    print(f"  [OK] Created {len(chunks)} semantic chunks")
    
    # Step 2: Generate Embeddings
    print("\nStep 2/4: Generating Embeddings...")
    embedder = EmbeddingGenerator()
    
    # Get texts with context injection
    texts = [chunk.get_injected_text() for chunk in chunks]
    embeddings = embedder.generate_embeddings(texts)
    
    print(f"  [OK] Generated {len(embeddings)} embeddings (dim: {len(embeddings[0]) if len(embeddings) > 0 else 0})")
    
    # Step 3: Store in Vector Database
    print("\nStep 3/4: Storing in Vector Database...")
    vector_store = VectorStore()
    
    # Prepare documents for storage
    ids = []
    texts = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(str(chunk.id))  # Convert to string for ChromaDB
        texts.append(chunk.get_injected_text())
        metadatas.append({
            **chunk.metadata,
            "source": chunk.source_file,
            "basin": chunk.detected_basin or "",
            "region": chunk.detected_region or "",
            "topics": json.dumps(chunk.topics) if chunk.topics else "[]",
            "context_summary": chunk.context_summary,
            "page_numbers": json.dumps(chunk.page_numbers) if chunk.page_numbers else "[]",
            "content": chunk.content,  # Original text without injection
        })
    
    vector_store.add_documents(ids, embeddings, texts, metadatas)
    print(f"  [OK] Stored {len(ids)} documents in ChromaDB")
    
    # Step 4: Build Knowledge Graph
    print("\nStep 4/4: Building Knowledge Graph...")
    kg_builder = KnowledgeGraphBuilder()
    
    for chunk in chunks:
        kg_builder.add_document(str(chunk.id), chunk.content, chunk.metadata)
    
    kg_builder.save()
    
    kg_stats = kg_builder.get_statistics()
    print(f"  [OK] Knowledge Graph: {kg_stats.get('nodes', 0)} nodes, {kg_stats.get('edges', 0)} edges")
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Documents processed: {len(set(c.source_file for c in chunks))}")
    print(f"Semantic chunks created: {len(chunks)}")
    print(f"Avg chunk size: {sum(c.metadata.get('char_count', 0) for c in chunks) / len(chunks):.0f} chars")
    print(f"Morocco basins detected: {len(set(c.detected_basin for c in chunks if c.detected_basin))}")
    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Knowledge graph nodes: {kg_stats.get('nodes', 0)}")
    print(f"Knowledge graph edges: {kg_stats.get('edges', 0)}")
    print(f"Total time: {elapsed:.1f} seconds")
    print("="*60)
    
    return {
        "chunks": len(chunks),
        "embeddings": len(embeddings),
        "kg_nodes": kg_stats.get('nodes', 0),
        "kg_edges": kg_stats.get('edges', 0),
        "elapsed_seconds": elapsed
    }


def main():
    parser = argparse.ArgumentParser(description="Re-ingest documents with semantic chunking")
    parser.add_argument(
        "--confirm", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--skip-clear",
        action="store_true", 
        help="Skip clearing existing data"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SEMANTIC RE-INGESTION PIPELINE")
    print("Advanced RAG System - Morocco Precision Agriculture")
    print("="*60)
    
    # Clear existing data
    if not args.skip_clear:
        if not clear_existing_data(confirm=args.confirm):
            return 1
    
    # Run ingestion
    try:
        stats = run_semantic_ingestion()
        print("\n[SUCCESS] Re-ingestion completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
