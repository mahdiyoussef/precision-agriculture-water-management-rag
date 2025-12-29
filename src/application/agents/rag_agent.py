"""
RAG Agent Module - Graph-Based Agentic RAG (GA-RAG)

Implements agentic retrieval with:
- Query Decomposition: Break complex queries into sub-problems
- Tool Selection: Route to Vector Search or Graph Traversal
- Iterative Synthesis: Autonomous context refinement
- Stateful Memory: Loop prevention during traversals
"""
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
import re

from src.config.config import LLM_CONFIG, logger

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Data Structures
# ============================================================================

class ToolType(Enum):
    """Available retrieval tools."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class QueryType(Enum):
    """Sub-query classification."""
    FACTUAL = "factual"
    RELATIONAL = "relational"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"


@dataclass
class SubQuery:
    """Decomposed sub-query with classification."""
    text: str
    query_type: QueryType
    tool: ToolType
    entities: List[str] = field(default_factory=list)
    resolved: bool = False
    context: str = ""


@dataclass
class AgentState:
    """
    Stateful memory for the agent to prevent infinite loops.
    Tracks visited entities and retrieval attempts.
    """
    visited_entities: Set[str] = field(default_factory=set)
    traversal_paths: List[List[str]] = field(default_factory=list)
    retrieval_attempts: int = 0
    max_attempts: int = 5
    context_buffer: List[str] = field(default_factory=list)
    
    def is_visited(self, entity: str) -> bool:
        return entity.lower() in self.visited_entities
    
    def mark_visited(self, entity: str):
        self.visited_entities.add(entity.lower())
    
    def can_continue(self) -> bool:
        return self.retrieval_attempts < self.max_attempts
    
    def increment_attempt(self):
        self.retrieval_attempts += 1
    
    def add_context(self, context: str):
        if context and context not in self.context_buffer:
            self.context_buffer.append(context)
    
    def get_full_context(self) -> str:
        return "\n\n".join(self.context_buffer)
    
    def reset(self):
        self.visited_entities.clear()
        self.traversal_paths.clear()
        self.retrieval_attempts = 0
        self.context_buffer.clear()


@dataclass
class AgentResult:
    """Result from the RAG Agent."""
    answer: str
    sub_queries: List[SubQuery]
    tools_used: List[str]
    hops_taken: int
    entities_visited: List[str]
    context_sources: List[str]
    confidence: float
    reasoning_trace: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sub_queries": [
                {"text": sq.text, "type": sq.query_type.value, "tool": sq.tool.value}
                for sq in self.sub_queries
            ],
            "tools_used": self.tools_used,
            "hops_taken": self.hops_taken,
            "entities_visited": self.entities_visited,
            "context_sources": self.context_sources,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace
        }


# ============================================================================
# RAG Agent
# ============================================================================

class RAGAgent:
    """
    Graph-Based Agentic RAG (GA-RAG) implementation.
    
    Orchestrates retrieval through:
    1. Phase 1: Query Decomposition - LLM-based decomposition
    2. Phase 2: Tool Selection - Vector vs Graph routing
    3. Phase 3: Iterative Synthesis - Context refinement
    
    Maintains stateful memory to prevent infinite traversal loops.
    """
    
    # Pattern-based query classification
    QUERY_PATTERNS = {
        QueryType.RELATIONAL: [
            r'\b(relationship|relation|connect|link|between|depend|affect|influence|cause|impact)\b',
            r'\b(how does .+ affect|what is the relationship|how are .+ related)\b',
        ],
        QueryType.COMPARATIVE: [
            r'\b(compare|differ|versus|vs\.|better|worse|advantage|disadvantage)\b',
            r'\b(which is (?:more|less|better)|difference between)\b',
        ],
        QueryType.PROCEDURAL: [
            r'\b(how to|steps to|process|procedure|method|technique|way to)\b',
            r'\b(implement|install|configure|set up|create)\b',
        ],
        QueryType.FACTUAL: [
            r'\b(what is|define|definition|meaning|explain|describe)\b',
            r'\b(when|where|who|which|how (?:much|many))\b',
        ],
    }
    
    def __init__(
        self,
        vector_store=None,
        knowledge_graph=None,
        hybrid_retriever=None,
        embedding_generator=None,
        llm_model: str = None,
        max_iterations: int = 5
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            vector_store: Vector store for semantic search
            knowledge_graph: Knowledge graph for relational queries
            hybrid_retriever: Combined retriever
            embedding_generator: Embedding model
            llm_model: LLM model name
            max_iterations: Maximum refinement iterations
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.hybrid_retriever = hybrid_retriever
        self.embedding_generator = embedding_generator
        self.llm_model = llm_model or LLM_CONFIG.get("model", "gemma2:9b")
        self.max_iterations = max_iterations
        
        # Agent state for loop prevention
        self.state = AgentState(max_attempts=max_iterations)
        
        # Initialize OpenAI client if available
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = openai.OpenAI(
                    base_url=LLM_CONFIG.get("base_url", "http://localhost:11434/v1"),
                    api_key=LLM_CONFIG.get("api_key", "ollama")
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        logger.info("RAG Agent initialized")
    
    # ========================================================================
    # Phase 1: Query Decomposition
    # ========================================================================
    
    def decompose_query(self, query: str) -> List[SubQuery]:
        """
        Decompose complex query into sub-problems.
        
        Uses LLM for decomposition with fallback to pattern-based.
        
        Args:
            query: Complex user query
            
        Returns:
            List of sub-queries with classifications
        """
        # Try LLM-based decomposition first
        if self.client:
            try:
                return self._llm_decompose(query)
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}")
        
        # Fallback to pattern-based
        return self._pattern_decompose(query)
    
    def _llm_decompose(self, query: str) -> List[SubQuery]:
        """LLM-based query decomposition."""
        prompt = f"""Break down this query into simpler sub-questions. For each sub-question, classify it as:
- FACTUAL: Seeking specific facts or definitions
- RELATIONAL: Seeking relationships or connections between concepts
- COMPARATIVE: Comparing two or more things
- PROCEDURAL: Seeking steps or methods

Query: {query}

Output format (one per line):
TYPE: Sub-question text | ENTITIES: entity1, entity2

Example:
FACTUAL: What is drip irrigation? | ENTITIES: drip irrigation
RELATIONAL: How does soil moisture affect irrigation scheduling? | ENTITIES: soil moisture, irrigation scheduling"""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        return self._parse_decomposition(content)
    
    def _parse_decomposition(self, content: str) -> List[SubQuery]:
        """Parse LLM decomposition output."""
        sub_queries = []
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse TYPE: text | ENTITIES: ...
            match = re.match(
                r'(FACTUAL|RELATIONAL|COMPARATIVE|PROCEDURAL):\s*(.+?)(?:\s*\|\s*ENTITIES:\s*(.+))?$',
                line,
                re.IGNORECASE
            )
            
            if match:
                query_type_str = match.group(1).upper()
                text = match.group(2).strip()
                entities_str = match.group(3) or ""
                
                query_type = QueryType[query_type_str]
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                
                # Determine tool based on query type
                if query_type == QueryType.RELATIONAL:
                    tool = ToolType.GRAPH
                elif query_type == QueryType.FACTUAL:
                    tool = ToolType.VECTOR
                else:
                    tool = ToolType.HYBRID
                
                sub_queries.append(SubQuery(
                    text=text,
                    query_type=query_type,
                    tool=tool,
                    entities=entities
                ))
        
        # If no sub-queries parsed, return original as single query
        if not sub_queries:
            query_type = self._classify_query(content)
            sub_queries.append(SubQuery(
                text=content,
                query_type=query_type,
                tool=self._get_tool_for_type(query_type),
                entities=self._extract_entities_simple(content)
            ))
        
        return sub_queries
    
    def _pattern_decompose(self, query: str) -> List[SubQuery]:
        """Pattern-based query decomposition fallback."""
        # Split on conjunctions
        parts = re.split(r'\s+(?:and|or|also|then|but)\s+', query, flags=re.IGNORECASE)
        
        # Also split on question marks for multiple questions
        all_parts = []
        for part in parts:
            sub_parts = re.split(r'\?\s*', part)
            all_parts.extend([p.strip() + "?" for p in sub_parts if p.strip()])
        
        if not all_parts:
            all_parts = [query]
        
        sub_queries = []
        for part in all_parts:
            query_type = self._classify_query(part)
            sub_queries.append(SubQuery(
                text=part,
                query_type=query_type,
                tool=self._get_tool_for_type(query_type),
                entities=self._extract_entities_simple(part)
            ))
        
        return sub_queries
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type using patterns."""
        query_lower = query.lower()
        
        for query_type, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.FACTUAL  # Default
    
    def _get_tool_for_type(self, query_type: QueryType) -> ToolType:
        """Map query type to retrieval tool."""
        mapping = {
            QueryType.FACTUAL: ToolType.VECTOR,
            QueryType.RELATIONAL: ToolType.GRAPH,
            QueryType.COMPARATIVE: ToolType.HYBRID,
            QueryType.PROCEDURAL: ToolType.VECTOR,
        }
        return mapping.get(query_type, ToolType.HYBRID)
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction using patterns."""
        entities = []
        
        # Domain-specific patterns
        patterns = [
            r'\b(drip irrigation|sprinkler irrigation|flood irrigation|micro-irrigation)\b',
            r'\b(soil moisture|water use efficiency|evapotranspiration|crop coefficient)\b',
            r'\b(wheat|maize|corn|tomato|citrus|olive|almond|cotton)\b',
            r'\b(IoT|precision agriculture|remote sensing|GPS|sensor)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([m.lower() for m in matches])
        
        return list(set(entities))
    
    # ========================================================================
    # Phase 2: Tool Selection & Routing
    # ========================================================================
    
    def route_query(self, query: str) -> Tuple[ToolType, float]:
        """
        Route query to optimal retrieval tool.
        
        Args:
            query: Query text
            
        Returns:
            Tuple of (ToolType, confidence)
        """
        query_type = self._classify_query(query)
        tool = self._get_tool_for_type(query_type)
        
        # Calculate confidence based on pattern match strength
        confidence = 0.7  # Base confidence
        
        query_lower = query.lower()
        for qt, patterns in self.QUERY_PATTERNS.items():
            if qt == query_type:
                match_count = sum(
                    1 for p in patterns 
                    if re.search(p, query_lower)
                )
                confidence = min(0.95, 0.7 + match_count * 0.1)
                break
        
        return tool, confidence
    
    def execute_tool(
        self,
        sub_query: SubQuery,
        top_k: int = 5
    ) -> Tuple[str, List[str]]:
        """
        Execute the appropriate retrieval tool for a sub-query.
        
        Args:
            sub_query: Sub-query with tool assignment
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (context, sources)
        """
        self.state.increment_attempt()
        
        if sub_query.tool == ToolType.VECTOR:
            return self._execute_vector_search(sub_query.text, top_k)
        elif sub_query.tool == ToolType.GRAPH:
            return self._execute_graph_search(sub_query, top_k)
        else:  # HYBRID
            return self._execute_hybrid_search(sub_query, top_k)
    
    def _execute_vector_search(
        self, 
        query: str, 
        top_k: int
    ) -> Tuple[str, List[str]]:
        """Execute vector semantic search."""
        if self.hybrid_retriever is None:
            return "", []
        
        try:
            docs = self.hybrid_retriever.retrieve(query, top_k=top_k)
            
            context_parts = []
            sources = []
            
            for doc in docs:
                text = doc.get("text", doc.get("content", ""))
                source = doc.get("source_file", doc.get("source", "unknown"))
                
                context_parts.append(text)
                sources.append(source)
            
            return "\n\n".join(context_parts), list(set(sources))
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return "", []
    
    def _execute_graph_search(
        self,
        sub_query: SubQuery,
        top_k: int
    ) -> Tuple[str, List[str]]:
        """Execute graph-based traversal search."""
        if self.knowledge_graph is None:
            return "", []
        
        try:
            context_parts = []
            sources = []
            
            # Use entities from sub-query or extract from text
            entities = sub_query.entities or self._extract_entities_simple(sub_query.text)
            
            for entity in entities[:3]:  # Limit to avoid explosion
                if self.state.is_visited(entity):
                    continue
                
                self.state.mark_visited(entity)
                
                # BFS traversal
                traversal = self.knowledge_graph.bfs_traverse(
                    entity, 
                    max_depth=2, 
                    max_nodes=top_k
                )
                
                if traversal.get("found"):
                    # Build context from traversal
                    context = f"Entity: {entity}\n"
                    for node in traversal.get("nodes", [])[:5]:
                        context += f"- {node['entity']} ({node['type']}) at depth {node['depth']}\n"
                    
                    for path in traversal.get("paths", [])[:3]:
                        context += f"Path: {' -> '.join(path['path'])}\n"
                    
                    context_parts.append(context)
                    
                    # Get related documents
                    docs = self.knowledge_graph.get_related_documents(entity)
                    sources.extend(docs[:3])
            
            # Also get KG context for the query
            kg_context = self.knowledge_graph.get_context_for_query(sub_query.text)
            if kg_context:
                context_parts.append(kg_context)
            
            return "\n\n".join(context_parts), list(set(sources))
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return "", []
    
    def _execute_hybrid_search(
        self,
        sub_query: SubQuery,
        top_k: int
    ) -> Tuple[str, List[str]]:
        """Execute hybrid search combining vector and graph."""
        # Get vector results
        vector_context, vector_sources = self._execute_vector_search(
            sub_query.text, 
            top_k // 2
        )
        
        # Get graph results
        graph_context, graph_sources = self._execute_graph_search(
            sub_query, 
            top_k // 2
        )
        
        # Combine contexts
        combined_context = ""
        if vector_context:
            combined_context += f"=== Semantic Search Results ===\n{vector_context}\n\n"
        if graph_context:
            combined_context += f"=== Graph Traversal Results ===\n{graph_context}"
        
        combined_sources = list(set(vector_sources + graph_sources))
        
        return combined_context, combined_sources
    
    # ========================================================================
    # Phase 3: Iterative Synthesis
    # ========================================================================
    
    def synthesize(
        self,
        query: str,
        sub_queries: List[SubQuery],
        contexts: List[str]
    ) -> Tuple[str, float]:
        """
        Synthesize final answer from sub-query contexts.
        
        Args:
            query: Original query
            sub_queries: Resolved sub-queries
            contexts: Retrieved contexts
            
        Returns:
            Tuple of (answer, confidence)
        """
        if not contexts or all(not c for c in contexts):
            return "Insufficient context to answer the query.", 0.0
        
        combined_context = "\n\n---\n\n".join([c for c in contexts if c])
        
        if self.client:
            try:
                return self._llm_synthesize(query, sub_queries, combined_context)
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
        
        # Fallback: return combined context
        return combined_context[:2000], 0.5
    
    def _llm_synthesize(
        self,
        query: str,
        sub_queries: List[SubQuery],
        context: str
    ) -> Tuple[str, float]:
        """LLM-based answer synthesis."""
        sub_q_text = "\n".join([
            f"- {sq.text} ({sq.query_type.value})" 
            for sq in sub_queries
        ])
        
        prompt = f"""Based on the context provided, answer the user's query.

Original Query: {query}

Sub-questions analyzed:
{sub_q_text}

Context:
{context[:4000]}

Provide a comprehensive answer that:
1. Directly addresses the query
2. Synthesizes information from all relevant sub-questions
3. Cites specific facts from the context
4. Acknowledges any gaps in information

Answer:"""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Estimate confidence based on context coverage
        confidence = 0.7
        if len(context) > 1000:
            confidence = 0.8
        if len(context) > 2000:
            confidence = 0.85
        
        return answer, confidence
    
    def check_context_sufficiency(
        self,
        query: str,
        context: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if context is sufficient to answer the query.
        
        Returns:
            Tuple of (is_sufficient, missing_entities)
        """
        if not context:
            return False, self._extract_entities_simple(query)
        
        # Simple heuristic: check if key entities appear in context
        entities = self._extract_entities_simple(query)
        context_lower = context.lower()
        
        missing = [e for e in entities if e.lower() not in context_lower]
        
        is_sufficient = len(missing) <= len(entities) // 2
        
        return is_sufficient, missing
    
    # ========================================================================
    # Main Execution
    # ========================================================================
    
    def run(
        self,
        query: str,
        top_k: int = 5
    ) -> AgentResult:
        """
        Execute the full agentic RAG pipeline.
        
        Args:
            query: User query
            top_k: Results per retrieval
            
        Returns:
            AgentResult with answer and metadata
        """
        # Reset state for new query
        self.state.reset()
        
        reasoning_trace = []
        tools_used = []
        
        # Phase 1: Query Decomposition
        reasoning_trace.append(f"Decomposing query: {query}")
        sub_queries = self.decompose_query(query)
        reasoning_trace.append(
            f"Identified {len(sub_queries)} sub-queries: " +
            ", ".join([sq.query_type.value for sq in sub_queries])
        )
        
        # Phase 2: Tool Selection & Execution
        all_contexts = []
        all_sources = []
        
        for sq in sub_queries:
            tool, confidence = self.route_query(sq.text)
            sq.tool = tool
            
            reasoning_trace.append(
                f"Routing '{sq.text[:50]}...' to {tool.value} (conf: {confidence:.2f})"
            )
            
            context, sources = self.execute_tool(sq, top_k)
            
            if context:
                sq.resolved = True
                sq.context = context
                all_contexts.append(context)
                all_sources.extend(sources)
                tools_used.append(tool.value)
                self.state.add_context(context)
        
        # Phase 3: Iterative Refinement (if needed)
        iteration = 0
        while self.state.can_continue() and iteration < self.max_iterations:
            is_sufficient, missing = self.check_context_sufficiency(
                query, 
                self.state.get_full_context()
            )
            
            if is_sufficient:
                reasoning_trace.append("Context sufficient for answer synthesis")
                break
            
            iteration += 1
            reasoning_trace.append(
                f"Iteration {iteration}: Refining for missing entities: {missing}"
            )
            
            # Try graph traversal for missing entities
            for entity in missing[:2]:
                if self.state.is_visited(entity):
                    continue
                
                refine_sq = SubQuery(
                    text=f"Information about {entity}",
                    query_type=QueryType.RELATIONAL,
                    tool=ToolType.GRAPH,
                    entities=[entity]
                )
                
                context, sources = self.execute_tool(refine_sq, top_k // 2)
                if context:
                    all_contexts.append(context)
                    all_sources.extend(sources)
                    tools_used.append("graph")
                    self.state.add_context(context)
        
        # Phase 4: Synthesis
        reasoning_trace.append("Synthesizing final answer")
        answer, confidence = self.synthesize(query, sub_queries, all_contexts)
        
        return AgentResult(
            answer=answer,
            sub_queries=sub_queries,
            tools_used=list(set(tools_used)),
            hops_taken=self.state.retrieval_attempts,
            entities_visited=list(self.state.visited_entities),
            context_sources=list(set(all_sources)),
            confidence=confidence,
            reasoning_trace=reasoning_trace
        )
    
    async def arun(
        self,
        query: str,
        top_k: int = 5
    ) -> AgentResult:
        """Async version of run for Python 3.12+ async support."""
        return await asyncio.to_thread(self.run, query, top_k)


# ============================================================================
# Factory Functions
# ============================================================================

def create_rag_agent(
    vector_store=None,
    knowledge_graph=None,
    hybrid_retriever=None,
    embedding_generator=None
) -> RAGAgent:
    """
    Factory function to create a RAG Agent.
    
    Args:
        vector_store: Vector store instance
        knowledge_graph: Knowledge graph instance
        hybrid_retriever: Hybrid retriever instance
        embedding_generator: Embedding generator instance
        
    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        hybrid_retriever=hybrid_retriever,
        embedding_generator=embedding_generator
    )


def main():
    """Test RAG Agent."""
    agent = RAGAgent()
    
    # Test query decomposition
    test_queries = [
        "How does drip irrigation affect soil moisture and what sensors can monitor it?",
        "Compare sprinkler and drip irrigation for water efficiency.",
        "What is evapotranspiration and how is it calculated?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        sub_queries = agent.decompose_query(query)
        for sq in sub_queries:
            print(f"  [{sq.query_type.value}] {sq.text}")
            print(f"    Tool: {sq.tool.value}")
            print(f"    Entities: {sq.entities}")
        
        tool, conf = agent.route_query(query)
        print(f"\n  Recommended tool: {tool.value} (confidence: {conf:.2f})")


if __name__ == "__main__":
    main()
