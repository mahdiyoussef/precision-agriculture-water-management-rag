"""
Query Enhancement Module
- Multi-query generation using LLM
- Domain-specific query expansion
- Query rewriting for better retrieval
"""
from typing import List, Dict, Optional
import re

from langchain_community.llms import Ollama

from ..config.config import LLM_CONFIG, RETRIEVAL_CONFIG, DOMAIN_SYNONYMS, logger


class QueryEnhancer:
    """
    Enhance queries for better retrieval:
    1. Multi-Query: Generate query variations
    2. Query Expansion: Add domain-specific synonyms
    3. Query Rewriting: Improve query clarity
    """
    
    def __init__(self, llm: Ollama = None):
        self.config = RETRIEVAL_CONFIG["query_enhancement"]
        self.multi_query_config = RETRIEVAL_CONFIG["multi_query"]
        
        # Initialize LLM for multi-query generation
        if llm is None:
            self.llm = Ollama(
                model=LLM_CONFIG["model"],
                base_url=LLM_CONFIG["base_url"],
                temperature=0.3,  # Slight creativity for query variations
            )
        else:
            self.llm = llm
        
        self.domain_synonyms = DOMAIN_SYNONYMS
    
    def generate_multi_queries(
        self,
        query: str,
        num_queries: int = None
    ) -> List[str]:
        """
        Generate query variations using LLM.
        
        Args:
            query: Original query
            num_queries: Number of variations to generate
        
        Returns:
            List of query variations (including original)
        """
        if not self.multi_query_config["enabled"]:
            return [query]
        
        num_queries = num_queries or self.multi_query_config["num_queries"]
        
        prompt = f"""You are an agricultural water management expert. Generate {num_queries} different versions of the following question to retrieve relevant documents about precision agriculture and water management.

Original question: {query}

Provide exactly {num_queries} alternative questions, one per line. Make each question approach the topic from a different angle while maintaining the core intent. Do not number the questions.

Alternative questions:"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse response into individual queries
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            # Clean up any numbering or bullet points
            cleaned_queries = []
            for line in lines:
                # Remove common prefixes
                line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
                if line and len(line) > 10:  # Skip very short lines
                    cleaned_queries.append(line)
            
            # Include original query first
            all_queries = [query] + cleaned_queries[:num_queries]
            
            logger.info(f"Generated {len(all_queries)} query variations")
            return all_queries
            
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return [query]
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with domain-specific synonyms.
        
        Args:
            query: Original query
        
        Returns:
            Expanded query with additional terms
        """
        if not self.config["expansion"]:
            return query
        
        query_lower = query.lower()
        expansion_terms = []
        
        for term, synonyms in self.domain_synonyms.items():
            if term.lower() in query_lower:
                # Add a few synonyms (not all, to avoid noise)
                expansion_terms.extend(synonyms[:2])
        
        if expansion_terms:
            # Deduplicate
            expansion_terms = list(set(expansion_terms))[:self.config["max_expansions"]]
            expanded = f"{query} {' '.join(expansion_terms)}"
            logger.debug(f"Expanded query: {expanded}")
            return expanded
        
        return query
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for clarity and better retrieval.
        
        Args:
            query: Original query
        
        Returns:
            Rewritten query
        """
        if not self.config["rewriting"]:
            return query
        
        prompt = f"""Rewrite the following question to be clearer and more specific for searching agricultural and water management documents. Keep the rewritten question concise but complete.

Original question: {query}

Rewritten question:"""
        
        try:
            response = self.llm.invoke(prompt)
            rewritten = response.strip()
            
            # Basic validation
            if len(rewritten) > 10 and len(rewritten) < 500:
                logger.debug(f"Rewritten query: {rewritten}")
                return rewritten
            else:
                return query
                
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query
    
    def enhance_query(
        self,
        query: str,
        use_multi_query: bool = True,
        use_expansion: bool = True,
        use_rewriting: bool = False
    ) -> Dict[str, any]:
        """
        Apply all query enhancement techniques.
        
        Args:
            query: Original query
            use_multi_query: Generate query variations
            use_expansion: Add synonyms
            use_rewriting: Rewrite for clarity
        
        Returns:
            Dictionary with enhanced queries
        """
        result = {
            "original": query,
            "expanded": query,
            "rewritten": query,
            "variations": [query],
        }
        
        # Expansion
        if use_expansion:
            result["expanded"] = self.expand_query(query)
        
        # Rewriting
        if use_rewriting:
            result["rewritten"] = self.rewrite_query(query)
        
        # Multi-query
        if use_multi_query:
            # Use expanded or rewritten query as base
            base_query = result["expanded"] if use_expansion else query
            result["variations"] = self.generate_multi_queries(base_query)
        
        return result
    
    def get_all_queries(
        self,
        query: str,
        include_original: bool = True
    ) -> List[str]:
        """
        Get all query variations for retrieval.
        
        Args:
            query: Original query
            include_original: Include the original query
        
        Returns:
            List of all query variations
        """
        enhanced = self.enhance_query(query)
        
        all_queries = set()
        
        if include_original:
            all_queries.add(query)
        
        all_queries.add(enhanced["expanded"])
        all_queries.update(enhanced["variations"])
        
        return list(all_queries)


def main():
    """Test query enhancement."""
    enhancer = QueryEnhancer()
    
    test_queries = [
        "How to save water in irrigation?",
        "What sensors are used in precision agriculture?",
        "Best practices for drip irrigation efficiency",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Original: {query}")
        
        enhanced = enhancer.enhance_query(query)
        
        print(f"Expanded: {enhanced['expanded']}")
        print(f"Variations:")
        for v in enhanced['variations']:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
