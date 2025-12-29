"""
Query Router Module
- Intelligent query classification
- Routes queries to optimal retrieval strategy
- Supports: factual, relational, summary, reasoning query types
"""
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from ..config.config import logger


class QueryIntent(Enum):
    """Query intent classification."""
    FACTUAL = "factual"           # "What is...", definition queries
    RELATIONAL = "relational"     # "How are X and Y related?"
    SUMMARY = "summary"           # "Summarize...", overview queries
    REASONING = "reasoning"       # Complex multi-step reasoning
    PROCEDURAL = "procedural"     # "How to...", step-by-step queries
    COMPARATIVE = "comparative"   # "Compare X and Y", "difference between"


@dataclass
class RetrievalStrategy:
    """Configuration for a retrieval strategy."""
    use_vector: bool = True
    use_keyword: bool = True
    use_graph: bool = False
    graph_depth: int = 1
    top_k: int = 5
    semantic_weight: float = 0.7
    use_reranking: bool = True
    use_query_expansion: bool = True


# Retrieval strategies for each intent type
INTENT_STRATEGIES: Dict[QueryIntent, RetrievalStrategy] = {
    QueryIntent.FACTUAL: RetrievalStrategy(
        use_vector=True,
        use_keyword=True,
        use_graph=False,
        top_k=5,
        semantic_weight=0.7,
        use_reranking=True,
        use_query_expansion=True,
    ),
    QueryIntent.RELATIONAL: RetrievalStrategy(
        use_vector=True,
        use_keyword=False,
        use_graph=True,
        graph_depth=2,
        top_k=7,
        semantic_weight=0.6,
        use_reranking=True,
        use_query_expansion=True,
    ),
    QueryIntent.SUMMARY: RetrievalStrategy(
        use_vector=True,
        use_keyword=False,
        use_graph=True,
        graph_depth=1,
        top_k=10,
        semantic_weight=0.8,
        use_reranking=True,
        use_query_expansion=False,
    ),
    QueryIntent.REASONING: RetrievalStrategy(
        use_vector=True,
        use_keyword=True,
        use_graph=True,
        graph_depth=2,
        top_k=8,
        semantic_weight=0.6,
        use_reranking=True,
        use_query_expansion=True,
    ),
    QueryIntent.PROCEDURAL: RetrievalStrategy(
        use_vector=True,
        use_keyword=True,
        use_graph=False,
        top_k=5,
        semantic_weight=0.7,
        use_reranking=True,
        use_query_expansion=True,
    ),
    QueryIntent.COMPARATIVE: RetrievalStrategy(
        use_vector=True,
        use_keyword=True,
        use_graph=True,
        graph_depth=1,
        top_k=8,
        semantic_weight=0.6,
        use_reranking=True,
        use_query_expansion=True,
    ),
}


class QueryRouter:
    """
    Routes queries to optimal retrieval strategy based on intent classification.
    
    Uses pattern matching and heuristics for fast, local classification.
    """
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        QueryIntent.FACTUAL: [
            r'^what\s+is\s+',
            r'^define\s+',
            r'^explain\s+what\s+',
            r'^what\s+are\s+the\s+(?:key\s+)?(?:features|characteristics|properties|types)\s+of',
            r'^what\s+does\s+.+\s+mean',
            r'^tell\s+me\s+about\s+',
        ],
        QueryIntent.RELATIONAL: [
            r'how\s+(?:is|are|does)\s+.+\s+(?:relate[ds]?\s+to|connected\s+to|linked\s+to|affect[s]?)',
            r'relationship\s+between',
            r'connection\s+between',
            r'how\s+does\s+.+\s+impact',
            r'what\s+is\s+the\s+(?:relation|connection|link)\s+between',
        ],
        QueryIntent.SUMMARY: [
            r'^summarize\s+',
            r'^give\s+(?:me\s+)?(?:a\s+)?(?:summary|overview)\s+',
            r'^provide\s+(?:a\s+)?(?:summary|overview)\s+',
            r'^what\s+is\s+the\s+(?:main|overall|general)\s+',
            r'^overview\s+of',
        ],
        QueryIntent.REASONING: [
            r'^why\s+(?:is|are|does|do|should)',
            r'^analyze\s+',
            r'^evaluate\s+',
            r'^what\s+would\s+happen\s+if',
            r'^what\s+are\s+the\s+(?:implications|consequences|effects)\s+of',
            r'design\s+.+\s+(?:system|architecture|plan)',
            r'create\s+.+\s+(?:plan|strategy|architecture)',
            r'develop\s+.+\s+(?:solution|approach)',
        ],
        QueryIntent.PROCEDURAL: [
            r'^how\s+(?:to|do\s+I|can\s+I|should\s+I)',
            r'^(?:steps|procedure|process)\s+(?:for|to)',
            r'^what\s+are\s+the\s+steps',
            r'^guide\s+(?:for|to|on)',
            r'implement\s+',
            r'install\s+',
            r'configure\s+',
            r'set\s*up\s+',
        ],
        QueryIntent.COMPARATIVE: [
            r'^compare\s+',
            r'^what\s+(?:is|are)\s+the\s+(?:difference|differences)\s+between',
            r'difference\s+between',
            r'vs\.?\s+',
            r'versus\s+',
            r'better\s+(?:than|between)',
            r'which\s+(?:is|are)\s+(?:better|best|preferred)',
        ],
    }
    
    # Keywords that hint at specific intents
    INTENT_KEYWORDS = {
        QueryIntent.FACTUAL: ["definition", "meaning", "what is", "explain"],
        QueryIntent.RELATIONAL: ["relationship", "connection", "impact", "effect", "influence"],
        QueryIntent.SUMMARY: ["summary", "overview", "brief", "main points"],
        QueryIntent.REASONING: ["why", "analyze", "evaluate", "implications", "design", "create", "architecture"],
        QueryIntent.PROCEDURAL: ["how to", "steps", "guide", "procedure", "implement"],
        QueryIntent.COMPARATIVE: ["compare", "difference", "versus", "vs", "better"],
    }
    
    def __init__(self):
        """Initialize the query router."""
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        logger.info("QueryRouter initialized")
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of a query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (QueryIntent, confidence_score)
        """
        query_lower = query.lower().strip()
        
        # Score each intent
        intent_scores: Dict[QueryIntent, float] = {intent: 0.0 for intent in QueryIntent}
        
        # Pattern matching (high weight)
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    intent_scores[intent] += 3.0
                    break
        
        # Keyword matching (medium weight)
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent_scores[intent] += 1.0
        
        # Additional heuristics
        
        # Long queries with multiple entities → likely relational or reasoning
        if len(query.split()) > 15:
            intent_scores[QueryIntent.REASONING] += 0.5
            intent_scores[QueryIntent.RELATIONAL] += 0.3
        
        # Questions with numbers → likely factual
        if re.search(r'\d+', query):
            intent_scores[QueryIntent.FACTUAL] += 0.3
        
        # Find best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Calculate confidence (normalize by max possible score ~5.0)
        confidence = min(max_score / 5.0, 1.0)
        
        # Default to REASONING if no clear pattern (catch-all for complex queries)
        if confidence < 0.2:
            best_intent = QueryIntent.REASONING
            confidence = 0.3
        
        logger.debug(f"Query intent: {best_intent.value} (confidence: {confidence:.2f})")
        return best_intent, confidence
    
    def get_strategy(self, query: str) -> Tuple[RetrievalStrategy, QueryIntent, float]:
        """
        Get the optimal retrieval strategy for a query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (RetrievalStrategy, QueryIntent, confidence)
        """
        intent, confidence = self.classify_intent(query)
        strategy = INTENT_STRATEGIES.get(intent, INTENT_STRATEGIES[QueryIntent.FACTUAL])
        
        logger.info(f"Query routed: intent={intent.value}, confidence={confidence:.2f}")
        return strategy, intent, confidence
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract key entities from query for graph search.
        
        Args:
            query: User query string
            
        Returns:
            List of entity strings
        """
        # Remove common stop words and question words
        stop_words = {
            "what", "is", "are", "the", "a", "an", "how", "to", "do", "does",
            "can", "could", "should", "would", "will", "for", "of", "in", "on",
            "with", "and", "or", "but", "about", "between", "from", "into",
            "through", "during", "before", "after", "above", "below", "i", "me",
            "my", "we", "our", "you", "your", "it", "its", "they", "them", "their",
            "this", "that", "these", "those", "be", "been", "being", "have", "has",
            "had", "having", "give", "tell", "create", "design", "make", "build",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        entities = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also extract noun phrases (simple heuristic: capitalized words)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend([w.lower() for w in capitalized])
        
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)
        
        return unique_entities[:10]  # Limit to 10 entities


def main():
    """Test query router."""
    router = QueryRouter()
    
    test_queries = [
        "What is drip irrigation?",
        "How does soil moisture relate to irrigation scheduling?",
        "Summarize the main water management techniques",
        "Create an irrigation architecture for 7 hectares of tomatoes",
        "How to install a soil moisture sensor?",
        "Compare drip irrigation and sprinkler systems",
        "Why is precision agriculture important for water conservation?",
    ]
    
    print("Query Router Test")
    print("=" * 60)
    
    for query in test_queries:
        strategy, intent, confidence = router.get_strategy(query)
        entities = router.extract_entities(query)
        
        print(f"\nQuery: {query}")
        print(f"  Intent: {intent.value} (confidence: {confidence:.2f})")
        print(f"  Strategy: vector={strategy.use_vector}, keyword={strategy.use_keyword}, graph={strategy.use_graph}")
        print(f"  Entities: {entities[:5]}")


if __name__ == "__main__":
    main()
