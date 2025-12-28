"""
Document Processing Pipeline
- PDF text extraction with table handling
- Smart semantic chunking
- Metadata extraction and tagging
- Chunk quality validation
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import pdfplumber
from tqdm import tqdm

from ..config.config import (
    DOCUMENTS_DIR, CHUNKS_DIR, METADATA_DIR, 
    CHUNK_CONFIG, logger
)


@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    chunk_id: str
    text: str
    source: str
    category: str
    page: int
    chunk_index: int
    total_chunks: int
    word_count: int
    char_count: int
    has_table: bool = False
    topics: List[str] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentProcessor:
    """Process PDF documents into chunks for the RAG system."""
    
    def __init__(self, documents_dir: Path = DOCUMENTS_DIR):
        self.documents_dir = documents_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CONFIG["chunk_size"],
            chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
            separators=CHUNK_CONFIG["separators"],
            length_function=CHUNK_CONFIG["length_function"],
            add_start_index=CHUNK_CONFIG["add_start_index"],
        )
        self.processed_chunks: List[DocumentChunk] = []
        logger.info(f"DocumentProcessor initialized with docs dir: {documents_dir}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level information.
        Uses pdfplumber for better table extraction.
        """
        pages = []
        
        try:
            # Try pdfplumber first for better table handling
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    has_table = len(tables) > 0
                    
                    # Convert tables to text if present
                    table_text = ""
                    if has_table:
                        for table in tables:
                            for row in table:
                                if row:
                                    # Filter None values and join
                                    row_text = " | ".join([str(cell) if cell else "" for cell in row])
                                    table_text += row_text + "\n"
                    
                    full_text = text
                    if table_text:
                        full_text += "\n\n[TABLE DATA]\n" + table_text
                    
                    pages.append({
                        "page_num": page_num,
                        "text": full_text,
                        "has_table": has_table
                    })
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}, falling back to pypdf: {e}")
            # Fallback to pypdf
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append({
                        "page_num": page_num,
                        "text": text,
                        "has_table": False
                    })
            except Exception as e2:
                logger.error(f"Failed to extract text from {pdf_path}: {e2}")
                return []
        
        return pages
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+\s*\|\s*Page', '', text)
        
        # Remove URLs (optional, might want to keep them)
        # text = re.sub(r'http[s]?://\S+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        return text.strip()
    
    def generate_chunk_id(self, source: str, page: int, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{source}_{page}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_category_from_path(self, pdf_path: Path) -> str:
        """Extract category from file path."""
        # Get parent folder name as category
        parent = pdf_path.parent.name
        if parent == "documents":
            return "uncategorized"
        return parent
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple keyword matching."""
        # Domain-specific topic keywords
        topic_keywords = {
            "irrigation": ["irrigation", "watering", "drip", "sprinkler", "flood irrigation"],
            "sensors": ["sensor", "IoT", "monitoring", "data collection", "smart"],
            "soil": ["soil", "moisture", "texture", "infiltration", "drainage"],
            "crops": ["crop", "plant", "agriculture", "farming", "yield"],
            "water_efficiency": ["efficiency", "conservation", "saving", "optimization"],
            "climate": ["climate", "weather", "temperature", "evapotranspiration", "rainfall"],
            "technology": ["technology", "precision", "automation", "digital", "AI", "machine learning"],
            "sustainability": ["sustainable", "environment", "ecosystem", "resource management"],
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_topics.append(topic)
                    break
        
        return list(set(found_topics))
    
    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a single PDF document into chunks."""
        source = pdf_path.name
        category = self.get_category_from_path(pdf_path)
        
        logger.info(f"Processing: {source}")
        
        # Extract pages
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            logger.warning(f"No content extracted from {source}")
            return []
        
        chunks = []
        chunk_index = 0
        
        for page_data in pages:
            page_num = page_data["page_num"]
            text = self.clean_text(page_data["text"])
            has_table = page_data["has_table"]
            
            if not text or len(text) < 50:  # Skip very short pages
                continue
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            for chunk_text in text_chunks:
                if len(chunk_text.strip()) < 50:  # Skip tiny chunks
                    continue
                
                chunk_id = self.generate_chunk_id(source, page_num, chunk_index)
                topics = self.extract_topics(chunk_text)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    category=category,
                    page=page_num,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update later
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    has_table=has_table,
                    topics=topics
                )
                chunks.append(chunk)
                chunk_index += 1
        
        # Update total_chunks for all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        logger.info(f"  Created {len(chunks)} chunks from {source}")
        return chunks
    
    def process_all_documents(self, save_chunks: bool = True) -> List[DocumentChunk]:
        """Process all PDF documents in the documents directory."""
        all_chunks = []
        
        # Find all PDFs recursively
        pdf_files = list(self.documents_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            chunks = self.process_document(pdf_path)
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        if save_chunks:
            self.save_chunks()
        
        return all_chunks
    
    def save_chunks(self, output_dir: Path = CHUNKS_DIR):
        """Save processed chunks to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all chunks in one file
        chunks_data = [chunk.to_dict() for chunk in self.processed_chunks]
        output_file = output_dir / "all_chunks.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks_data)} chunks to {output_file}")
        
        # Save metadata summary
        metadata = {
            "total_chunks": len(self.processed_chunks),
            "sources": list(set(c.source for c in self.processed_chunks)),
            "categories": list(set(c.category for c in self.processed_chunks)),
            "avg_chunk_length": sum(c.char_count for c in self.processed_chunks) / len(self.processed_chunks) if self.processed_chunks else 0,
        }
        
        metadata_file = METADATA_DIR / "processing_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load_chunks(self, input_dir: Path = CHUNKS_DIR) -> List[DocumentChunk]:
        """Load previously processed chunks."""
        input_file = input_dir / "all_chunks.json"
        
        if not input_file.exists():
            logger.warning(f"No chunks file found at {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.processed_chunks = [
            DocumentChunk(**chunk_data) for chunk_data in chunks_data
        ]
        
        logger.info(f"Loaded {len(self.processed_chunks)} chunks")
        return self.processed_chunks
    
    def get_chunks_by_category(self, category: str) -> List[DocumentChunk]:
        """Filter chunks by category."""
        return [c for c in self.processed_chunks if c.category == category]
    
    def get_chunks_by_source(self, source: str) -> List[DocumentChunk]:
        """Filter chunks by source document."""
        return [c for c in self.processed_chunks if c.source == source]


def main():
    """Test document processing."""
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Unique sources: {len(set(c.source for c in chunks))}")
    print(f"Categories: {set(c.category for c in chunks)}")
    
    if chunks:
        print(f"\nSample chunk:")
        sample = chunks[0]
        print(f"  ID: {sample.chunk_id}")
        print(f"  Source: {sample.source}")
        print(f"  Page: {sample.page}")
        print(f"  Length: {sample.char_count} chars")
        print(f"  Topics: {sample.topics}")
        print(f"  Text preview: {sample.text[:200]}...")


if __name__ == "__main__":
    main()
