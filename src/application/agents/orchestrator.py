"""
RAG Orchestrator Module - ReAct-style reasoning
- Explicit tool calls (vector_search, graph_search, keyword_search)
- Thought → Action → Observation → Critique loop
- Structured output with citations
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_community.llms import Ollama

from src.config.config import LLM_CONFIG, SYSTEM_PROMPT, logger
from src.retrieval.query_router import QueryRouter, QueryIntent


class ToolType(Enum):
    """Available retrieval tools."""
    VECTOR_SEARCH = "vector_search"
    GRAPH_SEARCH = "graph_search"
    KEYWORD_SEARCH = "keyword_search"
    DOCUMENT_SUMMARY = "get_document_summary"


@dataclass
class ToolCall:
    """Represents a tool call in the ReAct loop."""
    tool: ToolType
    params: Dict[str, Any]
    result: Optional[List[Dict]] = None


@dataclass
class ReActStep:
    """Single step in the ReAct reasoning loop."""
    thought: str
    action: Optional[ToolCall] = None
    observation: str = ""
    critique: str = ""


@dataclass
class OrchestratorResult:
    """Final result from the orchestrator."""
    executive_summary: str
    detailed_findings: List[Dict[str, str]]
    sources_table: List[Dict[str, Any]]
    raw_answer: str
    steps: List[ReActStep] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_markdown(self) -> str:
        """Format result as structured markdown."""
        md = []
        
        # Executive Summary
        md.append("## Executive Summary")
        md.append(self.executive_summary)
        md.append("")
        
        # Detailed Findings
        md.append("## Detailed Findings")
        for i, finding in enumerate(self.detailed_findings, 1):
            citation = finding.get("citation", "")
            text = finding.get("text", "")
            md.append(f"{i}. {text} {citation}")
        md.append("")
        
        # Sources Table
        md.append("## Sources")
        md.append("| # | Document | Relevance |")
        md.append("|---|----------|-----------|")
        for i, src in enumerate(self.sources_table, 1):
            doc = src.get("document", "Unknown")
            relevance = src.get("relevance", "Medium")
            md.append(f"| {i} | {doc} | {relevance} |")
        
        return "\n".join(md)


class RAGOrchestrator:
    """
    ReAct-style RAG Orchestrator.
    
    Implements the Thought → Action → Observation → Critique loop
    for intelligent, multi-step retrieval and reasoning.
    """
    
    # Tool selection prompt
    REACT_SYSTEM_PROMPT = """You are a RAG Orchestrator that uses a ReAct reasoning loop.

For each query, you must:
1. THOUGHT: Analyze what information is needed
2. ACTION: Choose and call a retrieval tool
3. OBSERVATION: Analyze the results
4. CRITIQUE: Decide if more information is needed

Available tools:
- vector_search(query, top_k): Semantic similarity search for concepts
- graph_search(entities, depth): Find relationships between entities
- keyword_search(query): Exact term matching for technical terms
- get_document_summary(doc_id): Get document overview

Output your reasoning in this format:
THOUGHT: [your analysis]
ACTION: [tool_name](param1, param2)
"""

    def __init__(
        self,
        hybrid_retriever,
        knowledge_graph,
        embedding_generator,
        memory_manager=None,
        max_iterations: int = 3
    ):
        """
        Initialize the orchestrator.
        
        Args:
            hybrid_retriever: HybridRetriever instance
            knowledge_graph: KnowledgeGraphBuilder instance
            embedding_generator: EmbeddingGenerator instance
            memory_manager: MemoryManager instance
            max_iterations: Max ReAct loop iterations
        """
        self.retriever = hybrid_retriever
        self.knowledge_graph = knowledge_graph
        self.embedder = embedding_generator
        self.memory = memory_manager
        self.max_iterations = max_iterations
        self.query_router = QueryRouter()
        
        # Initialize LLM for reasoning
        self.llm = Ollama(
            model=LLM_CONFIG["model"],
            base_url=LLM_CONFIG["base_url"],
            temperature=0.3,
            repeat_penalty=LLM_CONFIG["repeat_penalty"],
            stop=LLM_CONFIG.get("stop", []),
        )
        
        logger.info("RAGOrchestrator initialized with ReAct loop")
    
    # =========================================================================
    # RETRIEVAL TOOLS
    # =========================================================================
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic similarity search.
        Best for: Facts, definitions, conceptual queries.
        """
        logger.info(f"TOOL: vector_search('{query[:50]}...', top_k={top_k})")
        
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Format results
        formatted = []
        for doc in results:
            formatted.append({
                "id": doc.get("id", ""),
                "text": doc.get("text", "")[:500],  # Truncate for context
                "source": doc.get("metadata", {}).get("source", doc.get("source", "Unknown")),
                "score": doc.get("retrieval_score", 0),
                "tool": "vector_search",
            })
        
        return formatted
    
    def graph_search(self, entities: List[str], depth: int = 2) -> Dict[str, Any]:
        """
        Multi-hop entity relationship search.
        Best for: "How are X and Y related?", hierarchies, connections.
        """
        logger.info(f"TOOL: graph_search(entities={entities}, depth={depth})")
        
        graph_context = {
            "entities": [],
            "relationships": [],
            "context_text": "",
        }
        
        for entity in entities[:5]:  # Limit to 5 entities
            entity_info = self.knowledge_graph.query_entity(entity, max_depth=depth)
            if entity_info:
                graph_context["entities"].append(entity_info)
                
                # Extract relationships
                relations = entity_info.get("related_entities", [])
                for rel in relations[:10]:
                    graph_context["relationships"].append({
                        "from": entity,
                        "to": rel.get("entity", ""),
                        "type": rel.get("relationship", "related_to"),
                    })
        
        # Build context text
        if graph_context["entities"]:
            context_parts = []
            for ent in graph_context["entities"]:
                context_parts.append(f"Entity: {ent.get('entity', 'Unknown')}")
                for rel in ent.get("related_entities", [])[:5]:
                    context_parts.append(f"  - {rel.get('relationship', 'related_to')}: {rel.get('entity', '')}")
            graph_context["context_text"] = "\n".join(context_parts)
        
        return graph_context
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Exact keyword matching search.
        Best for: Technical terms, product IDs, specific terminology.
        """
        logger.info(f"TOOL: keyword_search('{query[:50]}...')")
        
        # Use BM25 component of hybrid retriever
        results = self.retriever.bm25_search(query, top_k=top_k)
        
        formatted = []
        for doc in results:
            formatted.append({
                "id": doc.get("id", ""),
                "text": doc.get("text", "")[:500],
                "source": doc.get("metadata", {}).get("source", doc.get("source", "Unknown")),
                "score": doc.get("bm25_score", 0),
                "tool": "keyword_search",
            })
        
        return formatted
    
    def get_document_summary(self, doc_id: str) -> str:
        """
        Get document-level summary.
        Best for: Big-picture context, understanding document scope.
        """
        logger.info(f"TOOL: get_document_summary(doc_id='{doc_id}')")
        
        # Try to get document from vector store
        doc = self.retriever.vector_store.get_document(doc_id)
        
        if doc:
            text = doc.get("text", doc.get("document", ""))
            source = doc.get("metadata", {}).get("source", "Unknown")
            
            # Create a brief summary (first 500 chars + context summary if available)
            context_summary = doc.get("metadata", {}).get("context_summary", "")
            if context_summary:
                return f"Document: {source}\nSummary: {context_summary}\n\nExcerpt: {text[:500]}..."
            else:
                return f"Document: {source}\n\nExcerpt: {text[:500]}..."
        
        return f"Document {doc_id} not found."
    
    # =========================================================================
    # REACT REASONING LOOP
    # =========================================================================
    
    def _select_tool(self, query: str, intent: QueryIntent) -> Tuple[ToolType, Dict[str, Any]]:
        """Select the best tool based on query intent."""
        
        # Extract entities for graph search
        entities = self.query_router.extract_entities(query)
        
        tool_mapping = {
            QueryIntent.FACTUAL: (ToolType.VECTOR_SEARCH, {"query": query, "top_k": 5}),
            QueryIntent.RELATIONAL: (ToolType.GRAPH_SEARCH, {"entities": entities, "depth": 2}),
            QueryIntent.SUMMARY: (ToolType.VECTOR_SEARCH, {"query": query, "top_k": 10}),
            QueryIntent.REASONING: (ToolType.VECTOR_SEARCH, {"query": query, "top_k": 8}),
            QueryIntent.PROCEDURAL: (ToolType.VECTOR_SEARCH, {"query": query, "top_k": 5}),
            QueryIntent.COMPARATIVE: (ToolType.VECTOR_SEARCH, {"query": query, "top_k": 8}),
        }
        
        return tool_mapping.get(intent, (ToolType.VECTOR_SEARCH, {"query": query, "top_k": 5}))
    
    def _execute_tool(self, tool: ToolType, params: Dict[str, Any]) -> Any:
        """Execute a retrieval tool."""
        if tool == ToolType.VECTOR_SEARCH:
            return self.vector_search(**params)
        elif tool == ToolType.GRAPH_SEARCH:
            return self.graph_search(**params)
        elif tool == ToolType.KEYWORD_SEARCH:
            return self.keyword_search(**params)
        elif tool == ToolType.DOCUMENT_SUMMARY:
            return self.get_document_summary(**params)
        else:
            logger.warning(f"Unknown tool: {tool}")
            return []
    
    def _build_context(self, tool_results: List[Any]) -> str:
        """Build context string from tool results."""
        context_parts = []
        
        for i, result in enumerate(tool_results):
            if isinstance(result, list):
                # Vector/keyword search results
                for doc in result:
                    source = doc.get("source", "Unknown")
                    text = doc.get("text", "")
                    context_parts.append(f"[Source {i+1}: {source}]\n{text}")
            elif isinstance(result, dict):
                # Graph search result
                context_parts.append(f"[Knowledge Graph Context]\n{result.get('context_text', '')}")
            elif isinstance(result, str):
                # Document summary
                context_parts.append(f"[Document Summary]\n{result}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def run(self, query: str) -> OrchestratorResult:
        """
        Run the ReAct orchestration loop.
        
        Args:
            query: User query
            
        Returns:
            OrchestratorResult with structured answer
        """
        logger.info(f"Orchestrator processing: {query[:50]}...")
        
        steps: List[ReActStep] = []
        all_results: List[Any] = []
        sources: List[Dict] = []
        
        # Step 1: Query Analysis
        strategy, intent, confidence = self.query_router.get_strategy(query)
        
        thought = f"Query intent: {intent.value} (confidence: {confidence:.2f}). "
        if intent == QueryIntent.RELATIONAL:
            thought += "This query asks about relationships between entities. Using graph search."
        elif intent == QueryIntent.FACTUAL:
            thought += "This is a factual query. Using vector search for semantic matching."
        elif intent == QueryIntent.REASONING:
            thought += "Complex reasoning required. Using vector search with more documents."
        else:
            thought += f"Using {intent.value} strategy."
        
        # Step 2: Execute primary tool
        tool, params = self._select_tool(query, intent)
        result = self._execute_tool(tool, params)
        all_results.append(result)
        
        step1 = ReActStep(
            thought=thought,
            action=ToolCall(tool=tool, params=params, result=result),
            observation=f"Retrieved {len(result) if isinstance(result, list) else 1} results"
        )
        steps.append(step1)
        
        # Track sources
        if isinstance(result, list):
            for doc in result:
                source = doc.get("source", "Unknown")
                if source not in [s["document"] for s in sources]:
                    sources.append({"document": source, "relevance": "High"})
        
        # Step 3: Check if we need graph context for complex queries
        if intent in [QueryIntent.RELATIONAL, QueryIntent.REASONING]:
            entities = self.query_router.extract_entities(query)
            if entities and tool != ToolType.GRAPH_SEARCH:
                graph_result = self.graph_search(entities[:3], depth=2)
                all_results.append(graph_result)
                
                step2 = ReActStep(
                    thought="Adding knowledge graph context for deeper understanding",
                    action=ToolCall(
                        tool=ToolType.GRAPH_SEARCH,
                        params={"entities": entities[:3], "depth": 2},
                        result=graph_result
                    ),
                    observation=f"Found {len(graph_result.get('entities', []))} entity connections"
                )
                steps.append(step2)
        
        # Step 4: Build context and generate answer
        context = self._build_context(all_results)
        kg_context = self.knowledge_graph.get_context_for_query(query)
        
        # Generate structured answer
        answer = self._generate_structured_answer(query, context, kg_context)
        
        # Parse the answer into structured format
        result = self._parse_structured_answer(answer, sources, steps)
        
        return result
    
    def _generate_structured_answer(
        self, 
        query: str, 
        context: str, 
        kg_context: str
    ) -> str:
        """Generate a structured answer using the LLM."""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}

You MUST format your response as follows:

## Executive Summary
[2-3 sentences summarizing the key answer]

## Detailed Findings
1. [First finding with citation] [Source: document_name]
2. [Second finding with citation] [Source: document_name]
3. [Additional findings...]

## Key Recommendations
- [Actionable recommendation 1]
- [Actionable recommendation 2]

If information is not in the context, state: "I cannot find this information in the provided documents."
<|eot_id|><|start_header_id|>user<|end_header_id|>

RETRIEVED CONTEXT:
{context}

KNOWLEDGE GRAPH CONTEXT:
{kg_context}

QUESTION: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {e}"
    
    def _parse_structured_answer(
        self,
        answer: str,
        sources: List[Dict],
        steps: List[ReActStep]
    ) -> OrchestratorResult:
        """Parse the LLM output into structured result."""
        
        # Extract sections
        executive_summary = ""
        detailed_findings = []
        
        # Try to extract executive summary
        exec_match = re.search(r'## Executive Summary\s*\n(.*?)(?=\n##|\Z)', answer, re.DOTALL)
        if exec_match:
            executive_summary = exec_match.group(1).strip()
        else:
            # Use first paragraph as summary
            lines = answer.strip().split('\n')
            executive_summary = lines[0] if lines else "No summary available."
        
        # Extract detailed findings
        findings_match = re.search(r'## Detailed Findings\s*\n(.*?)(?=\n##|\Z)', answer, re.DOTALL)
        if findings_match:
            findings_text = findings_match.group(1)
            # Parse numbered items
            items = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)', findings_text, re.DOTALL)
            for item in items:
                # Extract citation if present
                citation_match = re.search(r'\[Source:\s*([^\]]+)\]', item)
                citation = f"[Source: {citation_match.group(1)}]" if citation_match else ""
                text = re.sub(r'\[Source:[^\]]+\]', '', item).strip()
                detailed_findings.append({"text": text, "citation": citation})
        
        # Calculate confidence based on sources and findings
        confidence = min(0.4 + len(sources) * 0.1 + len(detailed_findings) * 0.05, 0.95)
        
        return OrchestratorResult(
            executive_summary=executive_summary,
            detailed_findings=detailed_findings,
            sources_table=sources[:10],  # Limit to 10 sources
            raw_answer=answer,
            steps=steps,
            confidence=confidence
        )


def main():
    """Test the orchestrator."""
    print("RAGOrchestrator module loaded successfully")
    print("Use with: orchestrator = RAGOrchestrator(retriever, kg, embedder)")


if __name__ == "__main__":
    main()
