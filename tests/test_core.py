"""
Unit Tests for Core Components

Tests for:
- ChunkingEvaluator metrics
- RAGAgent query decomposition and routing
- KnowledgeGraphBuilder graph traversal
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# ChunkingEvaluator Tests
# ============================================================================

class TestChunkingMetrics:
    """Test ChunkingMetrics dataclass."""
    
    def test_metrics_default_values(self):
        """Test default metric values."""
        from src.core.entities import ChunkingMetrics
        
        metrics = ChunkingMetrics()
        assert metrics.boundary_clarity == 0.0
        assert metrics.stickiness == 0.0
        assert metrics.cig_score == 0.0
    
    def test_metrics_summary(self):
        """Test summary string generation."""
        from src.core.entities import ChunkingMetrics
        
        metrics = ChunkingMetrics(
            boundary_clarity=0.75,
            stickiness=0.5,
            cig_score=0.8
        )
        summary = metrics.summary()
        
        assert "0.750" in summary
        assert "0.500" in summary
        assert "0.800" in summary
    
    def test_metrics_to_dict(self):
        """Test dictionary conversion."""
        from src.core.entities import ChunkingMetrics
        
        metrics = ChunkingMetrics(boundary_clarity=0.9)
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert d["boundary_clarity"] == 0.9


class TestChunkingEvaluator:
    """Test ChunkingEvaluator class."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        model = Mock()
        # Return fixed embeddings for testing
        model.encode = Mock(return_value=np.random.rand(5, 384))
        return model
    
    def test_evaluator_initialization(self, mock_embedding_model):
        """Test evaluator can be initialized with mock model."""
        from src.document_processing.semantic_chunker import ChunkingEvaluator
        
        evaluator = ChunkingEvaluator(
            embedding_model=mock_embedding_model,
            window_size=3,
            k_threshold=1.5
        )
        
        assert evaluator.window_size == 3
        assert evaluator.k_threshold == 1.5
    
    def test_evaluate_empty_chunks(self, mock_embedding_model):
        """Test evaluation with empty chunks returns default metrics."""
        from src.document_processing.semantic_chunker import ChunkingEvaluator
        
        evaluator = ChunkingEvaluator(embedding_model=mock_embedding_model)
        metrics = evaluator.evaluate_chunks([])
        
        assert metrics.boundary_clarity == 0.0
        assert metrics.stickiness == 0.0
    
    def test_cosine_similarity_static(self):
        """Test static cosine similarity calculation."""
        from src.document_processing.semantic_chunker import ChunkingEvaluator
        
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        
        sim = ChunkingEvaluator._cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.001  # Should be ~1.0 for identical vectors
        
        vec3 = np.array([0, 1, 0])
        sim_ortho = ChunkingEvaluator._cosine_similarity(vec1, vec3)
        assert abs(sim_ortho) < 0.001  # Should be ~0 for orthogonal vectors


# ============================================================================
# RAGAgent Tests
# ============================================================================

class TestQueryTypes:
    """Test query type enums and dataclasses."""
    
    def test_query_type_values(self):
        """Test QueryType enum values."""
        from src.core.entities import QueryType
        
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.RELATIONAL.value == "relational"
        assert QueryType.COMPARATIVE.value == "comparative"
    
    def test_tool_type_values(self):
        """Test ToolType enum values."""
        from src.core.entities import ToolType
        
        assert ToolType.VECTOR.value == "vector"
        assert ToolType.GRAPH.value == "graph"
        assert ToolType.HYBRID.value == "hybrid"


class TestAgentState:
    """Test AgentState class."""
    
    def test_state_initialization(self):
        """Test default state initialization."""
        from src.core.entities import AgentState
        
        state = AgentState()
        assert state.retrieval_attempts == 0
        assert len(state.visited_entities) == 0
    
    def test_mark_visited(self):
        """Test entity visit tracking."""
        from src.core.entities import AgentState
        
        state = AgentState()
        state.mark_visited("irrigation")
        state.mark_visited("IRRIGATION")  # Should normalize
        
        assert state.is_visited("irrigation")
        assert state.is_visited("IRRIGATION")
        assert len(state.visited_entities) == 1  # Deduped
    
    def test_can_continue(self):
        """Test attempt limit checking."""
        from src.core.entities import AgentState
        
        state = AgentState(max_attempts=3)
        
        assert state.can_continue()
        state.increment_attempt()
        state.increment_attempt()
        assert state.can_continue()
        state.increment_attempt()
        assert not state.can_continue()
    
    def test_context_buffer(self):
        """Test context accumulation."""
        from src.core.entities import AgentState
        
        state = AgentState()
        state.add_context("Context 1")
        state.add_context("Context 2")
        state.add_context("Context 1")  # Duplicate
        
        assert len(state.context_buffer) == 2
        full = state.get_full_context()
        assert "Context 1" in full
        assert "Context 2" in full
    
    def test_reset(self):
        """Test state reset."""
        from src.core.entities import AgentState
        
        state = AgentState()
        state.mark_visited("test")
        state.increment_attempt()
        state.add_context("ctx")
        
        state.reset()
        
        assert not state.is_visited("test")
        assert state.retrieval_attempts == 0
        assert len(state.context_buffer) == 0


class TestRAGAgent:
    """Test RAGAgent class."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized without dependencies."""
        from src.application.agents.rag_agent import RAGAgent
        
        agent = RAGAgent()
        assert agent.max_iterations == 5
    
    def test_query_classification_factual(self):
        """Test factual query classification."""
        from src.application.agents.rag_agent import RAGAgent
        from src.core.entities import QueryType
        
        agent = RAGAgent()
        qtype = agent._classify_query("What is drip irrigation?")
        
        assert qtype == QueryType.FACTUAL
    
    def test_query_classification_relational(self):
        """Test relational query classification."""
        from src.application.agents.rag_agent import RAGAgent
        from src.core.entities import QueryType
        
        agent = RAGAgent()
        qtype = agent._classify_query("How does soil moisture affect irrigation?")
        
        assert qtype == QueryType.RELATIONAL
    
    def test_query_classification_comparative(self):
        """Test comparative query classification."""
        from src.application.agents.rag_agent import RAGAgent
        from src.core.entities import QueryType
        
        agent = RAGAgent()
        qtype = agent._classify_query("Compare drip vs sprinkler irrigation")
        
        assert qtype == QueryType.COMPARATIVE
    
    def test_pattern_decomposition(self):
        """Test pattern-based query decomposition."""
        from src.application.agents.rag_agent import RAGAgent
        
        agent = RAGAgent()
        query = "What is drip irrigation and how does it affect water efficiency?"
        
        sub_queries = agent._pattern_decompose(query)
        
        assert len(sub_queries) >= 1
        assert all(sq.text for sq in sub_queries)
    
    def test_entity_extraction(self):
        """Test simple entity extraction."""
        from src.application.agents.rag_agent import RAGAgent
        
        agent = RAGAgent()
        entities = agent._extract_entities_simple(
            "Drip irrigation with soil moisture sensors"
        )
        
        assert "drip irrigation" in entities or "soil moisture" in entities


# ============================================================================
# KnowledgeGraphBuilder Tests
# ============================================================================

class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder class."""
    
    @pytest.fixture
    def kg_builder(self):
        """Create KG builder without SpaCy for speed."""
        from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
        return KnowledgeGraphBuilder(use_spacy=False)
    
    def test_entity_extraction_patterns(self, kg_builder):
        """Test pattern-based entity extraction."""
        text = "Drip irrigation uses soil moisture sensors for precision agriculture."
        entities = kg_builder.extract_entities_pattern(text)
        
        assert len(entities) > 0
        entity_types = [e["type"] for e in entities]
        assert "irrigation_method" in entity_types or "sensor" in entity_types
    
    def test_add_document(self, kg_builder):
        """Test adding document to graph."""
        kg_builder.add_document(
            doc_id="test_doc",
            text="Drip irrigation improves water efficiency in arid regions."
        )
        
        assert kg_builder.graph.number_of_nodes() > 0
    
    def test_bfs_traverse_not_found(self, kg_builder):
        """Test BFS with non-existent entity."""
        result = kg_builder.bfs_traverse("nonexistent_entity")
        
        assert result["found"] == False
    
    def test_bfs_traverse_found(self, kg_builder):
        """Test BFS with existing entity."""
        kg_builder.add_document("doc1", "Drip irrigation is efficient.")
        kg_builder.add_document("doc2", "Drip irrigation uses sensors.")
        
        result = kg_builder.bfs_traverse("drip irrigation", max_depth=2)
        
        assert result["found"] == True
        assert len(result["nodes"]) >= 1
    
    def test_pagerank_empty_graph(self, kg_builder):
        """Test PageRank on empty graph."""
        result = kg_builder.pagerank_entities()
        assert result == []
    
    def test_pagerank_with_entities(self, kg_builder):
        """Test PageRank with entities."""
        kg_builder.add_document("doc1", "Drip irrigation is efficient.")
        kg_builder.add_document("doc2", "Precision agriculture uses sensors.")
        kg_builder.add_document("doc3", "Sensors monitor soil moisture.")
        
        result = kg_builder.pagerank_entities(top_k=5)
        
        assert len(result) > 0
        assert all("pagerank" in r for r in result)
    
    def test_get_shortest_path_not_found(self, kg_builder):
        """Test shortest path with missing entities."""
        result = kg_builder.get_shortest_path("entity_a", "entity_b")
        
        assert result["found"] == False
        assert result["length"] == -1


# ============================================================================
# Integration Tests (require models - marked slow)
# ============================================================================

@pytest.mark.slow
class TestIntegration:
    """Integration tests that load actual models."""
    
    def test_chunking_evaluator_real_model(self):
        """Test ChunkingEvaluator with real embedding model."""
        from src.document_processing.semantic_chunker import ChunkingEvaluator
        
        evaluator = ChunkingEvaluator(window_size=3)
        
        chunks = [
            {
                "text": "Drip irrigation is efficient.",
                "sentences": ["Drip irrigation is efficient."]
            },
            {
                "text": "Sensors monitor soil moisture.",
                "sentences": ["Sensors monitor soil moisture."]
            }
        ]
        
        metrics = evaluator.evaluate_chunks(chunks)
        
        assert 0 <= metrics.boundary_clarity <= 1
        assert metrics.cig_score >= 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
