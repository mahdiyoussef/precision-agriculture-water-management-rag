"""
RAG Chain Module
- LangChain RAG pipeline with Ollama
- Custom prompts for agriculture domain
- Source citation formatting
- Confidence scoring
"""
from typing import List, Dict, Any, Optional, Tuple
import re

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from ..config.config import LLM_CONFIG, GENERATION_CONFIG, SYSTEM_PROMPT, CITATION_CONFIG, logger
from .memory_manager import MemoryManager


class RAGChain:
    """
    RAG chain combining retrieval with LLM generation.
    Optimized for Qwen 1.8B via Ollama.
    """
    
    def __init__(
        self,
        retriever=None,
        memory: MemoryManager = None
    ):
        self.retriever = retriever
        self.memory = memory or MemoryManager()
        
        # Initialize LLM
        logger.info(f"Initializing Ollama with model: {LLM_CONFIG['model']}")
        self.llm = Ollama(
            model=LLM_CONFIG["model"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
            top_p=LLM_CONFIG["top_p"],
            top_k=LLM_CONFIG["top_k"],
            num_ctx=LLM_CONFIG["num_ctx"],
            repeat_penalty=LLM_CONFIG["repeat_penalty"],
            stop=LLM_CONFIG.get("stop", []),  # Add stop tokens
        )
        
        # Build prompts
        self.qa_prompt = self._build_qa_prompt()
    
    def _build_qa_prompt(self) -> PromptTemplate:
        """Build the QA prompt template."""
        # Llama 3 format with structured output instructions
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

RESPONSE FORMAT:
1. Start with a brief Executive Summary (2-3 sentences)
2. Provide Detailed Findings with inline citations [Source: document_name]
3. End with key takeaways or recommendations
4. If information is not in the context, state: "I cannot find this in the provided documents."
<|eot_id|><|start_header_id|>user<|end_header_id|>

{conversation_history}

RETRIEVED CONTEXT:
{context}

KNOWLEDGE GRAPH CONTEXT:
{knowledge_graph_context}

QUESTION: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return PromptTemplate(
            template=template,
            input_variables=["system_prompt", "conversation_history", "context", "knowledge_graph_context", "question"]
        )
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Formatted context string with source info
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.get("metadata", {}).get("source", doc.get("source", "Unknown"))
            page = doc.get("metadata", {}).get("page", doc.get("page", "?"))
            category = doc.get("metadata", {}).get("category", doc.get("category", ""))
            text = doc.get("text", "")
            
            # Format source reference
            source_ref = f"[{i}] Source: {source}"
            if page != "?":
                source_ref += f", Page {page}"
            if category:
                source_ref += f" ({category})"
            
            context_parts.append(f"{source_ref}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_citations(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Generate citation list from documents.
        
        Args:
            documents: Retrieved documents
        
        Returns:
            List of formatted citations
        """
        citations = []
        
        for doc in documents[:CITATION_CONFIG["max_citations_per_response"]]:
            source = doc.get("metadata", {}).get("source", doc.get("source", "Unknown"))
            page = doc.get("metadata", {}).get("page", doc.get("page", None))
            
            citation = f"[Source: {source}"
            if page and CITATION_CONFIG["include_page_numbers"]:
                citation += f", page {page}"
            citation += "]"
            
            citations.append(citation)
        
        return citations
    
    def generate_response(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        knowledge_graph_context: str = "",
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response using RAG.
        
        Args:
            question: User question
            documents: Retrieved documents
            knowledge_graph_context: Context from knowledge graph
            use_memory: Whether to use conversation memory
        
        Returns:
            Dictionary with response and metadata
        """
        # Format context
        context = self.format_context(documents)
        
        # Get conversation history
        conversation_history = ""
        if use_memory and self.memory:
            conversation_history = self.memory.format_for_prompt()
        
        # Build prompt
        prompt = self.qa_prompt.format(
            system_prompt=SYSTEM_PROMPT,
            conversation_history=conversation_history,
            context=context,
            knowledge_graph_context=knowledge_graph_context,
            question=question
        )
        
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # Generate response
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = f"I apologize, but I encountered an error generating the response: {str(e)}"
        
        # Extract citations from response
        citations = self.format_citations(documents)
        
        # Calculate confidence (simple heuristic based on context relevance)
        confidence = self._estimate_confidence(response, documents)
        
        # Save to memory
        if use_memory and self.memory:
            self.memory.add_exchange(question, response)
        
        return {
            "answer": response,
            "sources": [doc.get("metadata", {}).get("source", doc.get("source")) for doc in documents],
            "citations": citations,
            "confidence": confidence,
            "num_sources": len(documents),
        }
    
    def _estimate_confidence(
        self,
        response: str,
        documents: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate confidence in the response.
        
        Simple heuristic based on:
        - Number of sources
        - Response length
        - Presence of uncertainty phrases
        """
        confidence = 0.5  # Base confidence
        
        # More sources = higher confidence
        num_docs = len(documents)
        if num_docs >= 3:
            confidence += 0.2
        elif num_docs >= 1:
            confidence += 0.1
        
        # Response has citations = higher confidence
        if "[Source:" in response:
            confidence += 0.1
        
        # Uncertainty phrases = lower confidence
        uncertainty_phrases = [
            "i don't know", "not sure", "unclear", "may not",
            "cannot find", "no information", "limited information"
        ]
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                confidence -= 0.15
                break
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return round(confidence, 2)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        include_kg_context: bool = True
    ) -> Dict[str, Any]:
        """
        Full RAG query pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_kg_context: Include knowledge graph context
        
        Returns:
            Response dictionary
        """
        if self.retriever is None:
            logger.error("No retriever configured")
            return {
                "answer": "Error: RAG system not fully initialized. Please run document ingestion first.",
                "sources": [],
                "citations": [],
                "confidence": 0.0,
                "num_sources": 0,
            }
        
        # Retrieve documents
        documents = self.retriever.retrieve(question, top_k=top_k)
        
        logger.info(f"Retrieved {len(documents)} documents for query")
        
        # Get knowledge graph context
        kg_context = ""
        if include_kg_context and hasattr(self.retriever, 'knowledge_graph'):
            kg_context = self.retriever.knowledge_graph.get_context_for_query(question)
        
        # Generate response
        return self.generate_response(
            question=question,
            documents=documents,
            knowledge_graph_context=kg_context,
            use_memory=True
        )
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()


def main():
    """Test RAG chain with mock data."""
    rag = RAGChain()
    
    # Mock documents
    mock_docs = [
        {
            "text": "Drip irrigation can achieve water use efficiency of 90-95% compared to 60-70% for traditional sprinkler systems. This makes it ideal for arid regions.",
            "metadata": {"source": "Irrigation_Efficiency.pdf", "page": 15}
        },
        {
            "text": "Key benefits of drip irrigation include reduced water consumption, lower energy costs, and improved crop yields due to precise water delivery.",
            "metadata": {"source": "Water_Conservation.pdf", "page": 23}
        },
    ]
    
    question = "What are the benefits of drip irrigation?"
    
    print(f"Question: {question}")
    print("-" * 50)
    
    result = rag.generate_response(question, mock_docs)
    
    print(f"Answer: {result['answer']}")
    print(f"\nConfidence: {result['confidence']}")
    print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
