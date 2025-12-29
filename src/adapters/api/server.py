"""
FastAPI REST API for RAG System

Provides endpoints for:
- /query - Single query
- /query/stream - Streaming query
- /health - Health check
- /info - System information

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
"""
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Pydantic Models with Examples
# ============================================================================

class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(
        ..., 
        description="User question to ask the RAG system",
        examples=["What are the best practices for drip irrigation?"]
    )
    top_k: int = Field(
        default=5, 
        description="Number of documents to retrieve",
        ge=1, 
        le=20
    )
    use_reranking: bool = Field(
        default=True, 
        description="Apply cross-encoder reranking for better precision"
    )
    use_agent: bool = Field(
        default=False, 
        description="Use agentic RAG with query decomposition and graph traversal"
    )
    stream: bool = Field(
        default=False, 
        description="Stream the response (use /query/stream endpoint instead)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How does drip irrigation affect water efficiency?",
                    "top_k": 5,
                    "use_reranking": True,
                    "use_agent": False
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str = Field(description="Generated answer from the RAG system")
    sources: List[str] = Field(default=[], description="Source documents used")
    confidence: float = Field(default=0.0, description="Confidence score (0-1)")
    query_time_ms: float = Field(default=0.0, description="Query processing time in milliseconds")
    tools_used: List[str] = Field(default=[], description="Retrieval tools used (vector, graph, hybrid)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="System status: healthy, degraded, or initializing")
    version: str = Field(default="2.0.0", description="API version")
    components: Dict[str, bool] = Field(default={}, description="Component health status")


class SystemInfo(BaseModel):
    """System information response."""
    vector_store_count: int = Field(default=0, description="Number of documents in vector store")
    knowledge_graph_nodes: int = Field(default=0, description="Number of entities in knowledge graph")
    knowledge_graph_edges: int = Field(default=0, description="Number of relationships in knowledge graph")
    embedding_model: str = Field(default="", description="Embedding model name")
    llm_model: str = Field(default="", description="LLM model name")


class EvaluationRequest(BaseModel):
    """Evaluation request model."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer to evaluate")
    context: List[str] = Field(..., description="Context chunks used for generation")


class EvaluationResponse(BaseModel):
    """Evaluation response model."""
    faithfulness: float = Field(description="Faithfulness score (0-1)")
    answer_relevancy: float = Field(description="Answer relevancy score (0-1)")
    step_efficiency: float = Field(description="Step efficiency for agentic queries (0-1)")
    overall_score: float = Field(description="Overall quality score (0-1)")
    verdict: str = Field(description="PASS, MARGINAL, or FAIL")


# ============================================================================
# FastAPI Application with Swagger
# ============================================================================

# API Tags for organization
tags_metadata = [
    {
        "name": "Query",
        "description": "Query the RAG system for agricultural water management information.",
    },
    {
        "name": "Evaluation",
        "description": "Evaluate RAG responses using LLM-as-Judge methodology.",
    },
    {
        "name": "System",
        "description": "System health and information endpoints.",
    },
]

app = FastAPI(
    title="Precision Agriculture RAG API",
    description="""
## Precision Agriculture Water Management RAG System

A Graph-Based Agentic RAG (GA-RAG) system for precision agriculture and water management.

### Features
- **Hybrid Search**: Semantic embeddings + BM25 keyword search
- **Knowledge Graph**: Entity relationships and multi-hop reasoning
- **Agentic RAG**: Query decomposition, tool selection, iterative synthesis
- **Streaming**: Server-Sent Events for real-time responses
- **Evaluation**: LLM-as-Judge with PPI calibration

### Authentication
Currently no authentication required (local deployment).

### Rate Limits
No rate limits for local deployment.
    """,
    version="2.0.0",
    contact={
        "name": "Precision Agriculture RAG",
        "url": "https://github.com/mahdiyoussef/precision-agriculture-water-management-rag",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Lazy load RAG system
_rag_system = None
_rag_agent = None


def get_rag_system():
    """Lazy load RAG system."""
    global _rag_system
    if _rag_system is None:
        from src.main import PrecisionAgricultureRAG
        _rag_system = PrecisionAgricultureRAG()
    return _rag_system


def get_rag_agent():
    """Lazy load RAG agent."""
    global _rag_agent
    if _rag_agent is None:
        from src.application.agents.rag_agent import RAGAgent
        rag = get_rag_system()
        _rag_agent = RAGAgent(
            hybrid_retriever=rag.hybrid_retriever,
            knowledge_graph=rag.knowledge_graph
        )
    return _rag_agent


# ============================================================================
# Endpoints
# ============================================================================

@app.get(
    "/health", 
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check the health status of the RAG system and its components."
)
async def health_check():
    """Check system health including vector store and knowledge graph."""
    components = {
        "rag_system": _rag_system is not None,
        "vector_store": False,
        "knowledge_graph": False,
    }
    
    if _rag_system:
        try:
            components["vector_store"] = _rag_system.vector_store.count() > 0
        except:
            pass
        try:
            components["knowledge_graph"] = _rag_system.knowledge_graph.graph.number_of_nodes() > 0
        except:
            pass
    
    return HealthResponse(
        status="healthy" if any(components.values()) else "initializing",
        components=components
    )


@app.get(
    "/info", 
    response_model=SystemInfo,
    tags=["System"],
    summary="System Information",
    description="Get detailed system information including document counts and model configurations."
)
async def system_info():
    """Get system statistics and configuration details."""
    rag = get_rag_system()
    
    try:
        vector_count = rag.vector_store.count()
    except:
        vector_count = 0
    
    try:
        kg_nodes = rag.knowledge_graph.graph.number_of_nodes()
        kg_edges = rag.knowledge_graph.graph.number_of_edges()
    except:
        kg_nodes = kg_edges = 0
    
    from src.config.config import EMBEDDING_CONFIG, LLM_CONFIG
    
    return SystemInfo(
        vector_store_count=vector_count,
        knowledge_graph_nodes=kg_nodes,
        knowledge_graph_edges=kg_edges,
        embedding_model=EMBEDDING_CONFIG["model_name"],
        llm_model=LLM_CONFIG["model"]
    )


@app.post(
    "/query", 
    response_model=QueryResponse,
    tags=["Query"],
    summary="Query RAG System",
    description="""
    Query the RAG system for agricultural water management information.
    
    **Standard Mode**: Uses hybrid retrieval (semantic + keyword search) with optional reranking.
    
    **Agentic Mode**: When `use_agent=true`, uses advanced query decomposition, 
    automatic tool selection (vector/graph/hybrid), and iterative refinement.
    """
)
async def query(request: QueryRequest):
    """Execute a query against the RAG system."""
    import time
    start = time.time()
    
    try:
        if request.use_agent:
            agent = get_rag_agent()
            result = await asyncio.to_thread(
                agent.run,
                request.question,
                request.top_k
            )
            
            return QueryResponse(
                answer=result.answer,
                sources=result.context_sources,
                confidence=result.confidence,
                query_time_ms=(time.time() - start) * 1000,
                tools_used=result.tools_used
            )
        else:
            rag = get_rag_system()
            result = await asyncio.to_thread(
                rag.query,
                request.question,
                request.top_k,
                request.use_reranking
            )
            
            return QueryResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                confidence=result.get("confidence", 0.0),
                query_time_ms=(time.time() - start) * 1000,
                tools_used=["hybrid_retriever"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query/stream",
    tags=["Query"],
    summary="Stream Query Response",
    description="""
    Stream the response using Server-Sent Events (SSE).
    
    **Event Types**:
    - `start`: Query processing started
    - `chunk`: Partial response text
    - `end`: Response complete with sources
    - `error`: Error occurred
    
    **Usage**: Connect to this endpoint with an SSE client.
    """
)
async def query_stream(request: QueryRequest):
    """Stream the RAG response in real-time."""
    async def generate():
        try:
            rag = get_rag_system()
            
            # Get context first
            from src.retrieval.hybrid_retriever import HybridRetriever
            docs = rag.hybrid_retriever.retrieve(request.question, top_k=request.top_k)
            
            # Format context
            context = "\n\n".join([d.get("text", "")[:500] for d in docs[:3]])
            
            # Stream from LLM
            from langchain_community.llms import Ollama
            from src.config.config import LLM_CONFIG
            
            llm = Ollama(
                model=LLM_CONFIG["model"],
                base_url=LLM_CONFIG["base_url"],
            )
            
            prompt = f"""Based on the context, answer the question.

Context:
{context}

Question: {request.question}

Answer:"""
            
            # Yield chunks
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01)
            
            # Final message with sources
            sources = [d.get("source_file", "unknown") for d in docs[:3]]
            yield f"data: {json.dumps({'type': 'end', 'sources': sources})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post(
    "/evaluate",
    tags=["Evaluation"],
    summary="Evaluate RAG Response",
    description="""
    Evaluate a RAG response using LLM-as-Judge methodology.
    
    **Metrics**:
    - **Faithfulness**: Are claims grounded in context?
    - **Answer Relevancy**: Does the answer address the question?
    - **Step Efficiency**: For agentic queries, how efficient was the path?
    """
)
async def evaluate_response(
    question: str = Body(..., description="Original question"),
    answer: str = Body(..., description="Generated answer to evaluate"),
    context: List[str] = Body(..., description="Context chunks used")
):
    """Evaluate RAG response quality using LLM-as-Judge."""
    try:
        from src.application.evaluation.agentic_eval import AgenticEvaluator
        
        evaluator = AgenticEvaluator()
        metrics = await asyncio.to_thread(
            evaluator.evaluate,
            question,
            answer,
            context
        )
        
        return metrics.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
