"""
Knowledge Graph Module
- Entity extraction using SpaCy NER + custom patterns
- Relationship extraction via dependency parsing
- NetworkX graph storage and querying
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import re

import networkx as nx

from ..config.config import KNOWLEDGE_GRAPH_CONFIG, KNOWLEDGE_GRAPH_DIR, logger


# Agricultural domain patterns for entity extraction
AGRICULTURAL_PATTERNS = {
    "irrigation_method": [
        r'\b(drip irrigation|flood irrigation|sprinkler irrigation|micro-irrigation|surface irrigation|subsurface irrigation|center pivot|furrow irrigation|basin irrigation)\b',
        r'\b(irrigation system|irrigation method|irrigation technique)\b',
    ],
    "crop": [
        r'\b(wheat|rice|corn|maize|cotton|sugarcane|soybean|coffee|tea|potato|tomato|citrus|grape|olive|almond|alfalfa)\b',
        r'\b(crop|crops|plant|plants|vegetation|cultivar|variety)\b',
    ],
    "sensor": [
        r'\b(soil moisture sensor|tensiometer|TDR|time domain reflectometry|capacitance sensor|neutron probe)\b',
        r'\b(sensor|probe|monitor|detector|meter|gauge)\b',
    ],
    "technology": [
        r'\b(precision agriculture|smart farming|IoT|internet of things|remote sensing|GIS|GPS|drone|UAV|satellite imagery)\b',
        r'\b(machine learning|AI|artificial intelligence|automation|data analytics)\b',
    ],
    "water_source": [
        r'\b(groundwater|surface water|river|lake|reservoir|aquifer|well|rainwater|recycled water)\b',
    ],
    "metric": [
        r'\b(evapotranspiration|ET0|ETc|water use efficiency|WUE|irrigation efficiency|crop coefficient|Kc)\b',
        r'\b(soil moisture|moisture content|water potential|matric potential|field capacity|wilting point)\b',
    ],
    "region": [
        r'\b(arid|semi-arid|Mediterranean|tropical|subtropical|temperate)\b',
    ],
    "organization": [
        r'\b(FAO|USDA|ICARDA|IWMI|CGIAR|EPA)\b',
    ],
}


class KnowledgeGraphBuilder:
    """
    Build and query knowledge graph from documents.
    
    Extracts entities and relationships to provide
    additional context for RAG retrieval.
    """
    
    def __init__(self, use_spacy: bool = True):
        self.config = KNOWLEDGE_GRAPH_CONFIG
        self.graph = nx.MultiDiGraph()
        self.entity_index = defaultdict(set)  # entity -> document IDs
        self.entity_frequency = defaultdict(int)
        
        # Try to load SpaCy
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(self.config["extraction_model"])
                logger.info(f"Loaded SpaCy model: {self.config['extraction_model']}")
            except Exception as e:
                logger.warning(f"Failed to load SpaCy: {e}. Using pattern-based extraction only.")
        
        # Compile patterns
        self.patterns = {}
        for entity_type, pattern_list in AGRICULTURAL_PATTERNS.items():
            combined = '|'.join(f'({p})' for p in pattern_list)
            self.patterns[entity_type] = re.compile(combined, re.IGNORECASE)
    
    def extract_entities_pattern(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity_text = match.group().strip().lower()
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "method": "pattern"
                })
        
        return entities
    
    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using SpaCy NER."""
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        # Map SpaCy labels to our entity types
        label_mapping = {
            "ORG": "organization",
            "GPE": "region",
            "LOC": "region",
            "PRODUCT": "technology",
            "QUANTITY": "metric",
            "PERCENT": "metric",
        }
        
        for ent in doc.ents:
            entity_type = label_mapping.get(ent.label_, None)
            if entity_type:
                entities.append({
                    "text": ent.text.lower(),
                    "type": entity_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "spacy"
                })
        
        return entities
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using both patterns and SpaCy."""
        pattern_entities = self.extract_entities_pattern(text)
        spacy_entities = self.extract_entities_spacy(text)
        
        # Combine and deduplicate
        all_entities = pattern_entities + spacy_entities
        
        # Deduplicate by text and type
        seen = set()
        unique_entities = []
        for entity in all_entities:
            key = (entity["text"], entity["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        Uses co-occurrence and dependency parsing.
        """
        relationships = []
        
        # Simple co-occurrence within sentences
        if self.nlp:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            for sent in sentences:
                sent_text = sent.text.lower()
                entities_in_sent = [
                    e for e in entities 
                    if e["text"] in sent_text
                ]
                
                # Create co-occurrence relationships
                for i, e1 in enumerate(entities_in_sent):
                    for e2 in entities_in_sent[i+1:]:
                        if e1["type"] != e2["type"]:
                            relationships.append({
                                "subject": e1["text"],
                                "subject_type": e1["type"],
                                "predicate": "related_to",
                                "object": e2["text"],
                                "object_type": e2["type"],
                            })
        else:
            # Fallback: simple co-occurrence in text
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    if e1["type"] != e2["type"]:
                        # Check if they're close in text
                        distance = abs(e1["start"] - e2["start"])
                        if distance < 500:  # Within 500 chars
                            relationships.append({
                                "subject": e1["text"],
                                "subject_type": e1["type"],
                                "predicate": "related_to",
                                "object": e2["text"],
                                "object_type": e2["type"],
                            })
        
        return relationships
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Process a document and add to knowledge graph.
        
        Args:
            doc_id: Document identifier
            text: Document text
            metadata: Document metadata
        """
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities)
        
        # Add entities as nodes
        for entity in entities:
            entity_text = entity["text"]
            entity_type = entity["type"]
            
            # Track frequency
            self.entity_frequency[entity_text] += 1
            
            # Add to index
            self.entity_index[entity_text].add(doc_id)
            
            # Add node if not exists
            if not self.graph.has_node(entity_text):
                self.graph.add_node(
                    entity_text,
                    type=entity_type,
                    frequency=1,
                    documents=set([doc_id])
                )
            else:
                # Update existing node
                self.graph.nodes[entity_text]["frequency"] += 1
                self.graph.nodes[entity_text]["documents"].add(doc_id)
        
        # Add relationships as edges
        for rel in relationships:
            self.graph.add_edge(
                rel["subject"],
                rel["object"],
                relation=rel["predicate"],
                source=doc_id
            )
    
    def build_from_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Build knowledge graph from document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'text'
        """
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        
        for chunk in chunks:
            self.add_document(
                doc_id=chunk.get("chunk_id", chunk.get("id")),
                text=chunk["text"],
                metadata=chunk.get("metadata")
            )
        
        # Filter low-frequency entities
        min_freq = self.config["min_entity_frequency"]
        nodes_to_remove = [
            node for node, freq in self.entity_frequency.items()
            if freq < min_freq
        ]
        self.graph.remove_nodes_from(nodes_to_remove)
        
        logger.info(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def query_entity(
        self,
        entity: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Query the graph for information about an entity.
        
        Args:
            entity: Entity text to query
            max_depth: Maximum traversal depth
        
        Returns:
            Dictionary with entity information
        """
        entity = entity.lower()
        
        if entity not in self.graph:
            # Try partial match
            matches = [
                n for n in self.graph.nodes() 
                if entity in n or n in entity
            ]
            if matches:
                entity = matches[0]
            else:
                return {"found": False, "entity": entity}
        
        # Get ego graph (subgraph around entity)
        subgraph = nx.ego_graph(self.graph, entity, radius=max_depth)
        
        # Get direct neighbors
        neighbors = list(self.graph.neighbors(entity))
        predecessors = list(self.graph.predecessors(entity))
        
        # Get node data
        node_data = self.graph.nodes[entity]
        
        # Get relationships
        relationships = []
        for u, v, data in self.graph.edges(entity, data=True):
            relationships.append({
                "from": u,
                "to": v,
                "relation": data.get("relation", "related_to")
            })
        for u, v, data in self.graph.in_edges(entity, data=True):
            relationships.append({
                "from": u,
                "to": v,
                "relation": data.get("relation", "related_to")
            })
        
        return {
            "found": True,
            "entity": entity,
            "type": node_data.get("type"),
            "frequency": node_data.get("frequency", 0),
            "documents": list(node_data.get("documents", set())),
            "related_entities": list(set(neighbors + predecessors)),
            "relationships": relationships,
            "subgraph_size": len(subgraph.nodes()),
        }
    
    def get_context_for_query(
        self,
        query: str,
        max_entities: int = 5
    ) -> str:
        """
        Extract relevant knowledge graph context for a query.
        
        Args:
            query: Query text
            max_entities: Maximum number of entities to include
        
        Returns:
            Context string from knowledge graph
        """
        # Extract entities from query
        query_entities = self.extract_entities(query)
        
        if not query_entities:
            return ""
        
        context_parts = []
        
        for entity in query_entities[:max_entities]:
            entity_text = entity["text"]
            info = self.query_entity(entity_text)
            
            if info["found"]:
                context = f"- {entity_text} ({info['type']})"
                if info["related_entities"]:
                    related = ", ".join(info["related_entities"][:5])
                    context += f" is related to: {related}"
                context_parts.append(context)
        
        if context_parts:
            return "Knowledge Graph Context:\n" + "\n".join(context_parts)
        return ""
    
    def get_related_documents(
        self,
        entity: str
    ) -> List[str]:
        """Get document IDs containing an entity."""
        entity = entity.lower()
        return list(self.entity_index.get(entity, set()))
    
    def save(self, filepath: Path = None):
        """Save knowledge graph to disk."""
        if filepath is None:
            filepath = KNOWLEDGE_GRAPH_DIR / "knowledge_graph.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for serialization
        graph_data = {
            "graph": nx.node_link_data(self.graph),
            "entity_index": {k: list(v) for k, v in self.entity_index.items()},
            "entity_frequency": dict(self.entity_frequency),
        }
        
        # Handle node documents sets
        for node_data in graph_data["graph"]["nodes"]:
            if "documents" in node_data:
                node_data["documents"] = list(node_data["documents"])
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_data, f)
        
        logger.info(f"Saved knowledge graph to {filepath}")
    
    def load(self, filepath: Path = None):
        """Load knowledge graph from disk."""
        if filepath is None:
            filepath = KNOWLEDGE_GRAPH_DIR / "knowledge_graph.pkl"
        
        if not filepath.exists():
            logger.warning(f"No knowledge graph found at {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Convert documents lists back to sets
        for node_data in graph_data["graph"]["nodes"]:
            if "documents" in node_data:
                node_data["documents"] = set(node_data["documents"])
        
        self.graph = nx.node_link_graph(graph_data["graph"])
        self.entity_index = defaultdict(set, {
            k: set(v) for k, v in graph_data["entity_index"].items()
        })
        self.entity_frequency = defaultdict(int, graph_data["entity_frequency"])
        
        logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    # ========================================================================
    # Graph Traversal Methods for Agentic GraphRAG
    # ========================================================================
    
    def bfs_traverse(
        self,
        start_entity: str,
        max_depth: int = 2,
        max_nodes: int = 20
    ) -> Dict[str, Any]:
        """
        Breadth-first traversal from a starting entity.
        Used for relational multi-hop reasoning in GraphRAG.
        
        Args:
            start_entity: Starting entity for traversal
            max_depth: Maximum traversal depth (hops)
            max_nodes: Maximum nodes to return
            
        Returns:
            Dictionary with traversal results and path information
        """
        start_entity = start_entity.lower()
        
        if start_entity not in self.graph:
            # Try partial match
            matches = [n for n in self.graph.nodes() if start_entity in n or n in start_entity]
            if matches:
                start_entity = matches[0]
            else:
                return {"found": False, "entity": start_entity, "nodes": [], "paths": []}
        
        visited = {start_entity: 0}  # node -> depth
        queue = [(start_entity, 0, [start_entity])]  # (node, depth, path)
        result_nodes = [start_entity]
        paths = []
        
        while queue and len(result_nodes) < max_nodes:
            current, depth, path = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors (both directions for multi-hop)
            neighbors = set(self.graph.successors(current)) | set(self.graph.predecessors(current))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    new_path = path + [neighbor]
                    queue.append((neighbor, depth + 1, new_path))
                    result_nodes.append(neighbor)
                    paths.append({
                        "path": new_path,
                        "length": len(new_path) - 1,
                        "end_entity": neighbor,
                        "end_type": self.graph.nodes[neighbor].get("type", "unknown")
                    })
                    
                    if len(result_nodes) >= max_nodes:
                        break
        
        # Get node details
        nodes_with_data = []
        for node in result_nodes:
            node_data = self.graph.nodes[node]
            nodes_with_data.append({
                "entity": node,
                "type": node_data.get("type", "unknown"),
                "depth": visited[node],
                "frequency": node_data.get("frequency", 0)
            })
        
        return {
            "found": True,
            "start_entity": start_entity,
            "nodes": nodes_with_data,
            "paths": paths[:10],  # Top 10 paths
            "total_discovered": len(result_nodes)
        }
    
    def pagerank_entities(
        self,
        top_k: int = 20,
        damping: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Calculate PageRank scores for entities in the knowledge graph.
        Higher scores indicate more important/central entities.
        
        Args:
            top_k: Number of top entities to return
            damping: PageRank damping factor
            
        Returns:
            List of entities with PageRank scores
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        try:
            # Calculate PageRank
            pagerank_scores = nx.pagerank(self.graph, alpha=damping)
            
            # Sort by score
            sorted_entities = sorted(
                pagerank_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            # Add node metadata
            result = []
            for entity, score in sorted_entities:
                node_data = self.graph.nodes.get(entity, {})
                result.append({
                    "entity": entity,
                    "pagerank": round(score, 6),
                    "type": node_data.get("type", "unknown"),
                    "frequency": node_data.get("frequency", 0),
                    "connections": self.graph.degree(entity)
                })
            
            return result
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            return []
    
    def get_shortest_path(
        self,
        entity_a: str,
        entity_b: str
    ) -> Dict[str, Any]:
        """
        Find shortest path between two entities.
        Used for Step Efficiency metric in agentic evaluation.
        
        Args:
            entity_a: Source entity
            entity_b: Target entity
            
        Returns:
            Dictionary with path information
        """
        entity_a = entity_a.lower()
        entity_b = entity_b.lower()
        
        # Find matching nodes
        def find_node(entity: str) -> Optional[str]:
            if entity in self.graph:
                return entity
            matches = [n for n in self.graph.nodes() if entity in n or n in entity]
            return matches[0] if matches else None
        
        node_a = find_node(entity_a)
        node_b = find_node(entity_b)
        
        if node_a is None or node_b is None:
            return {
                "found": False,
                "source": entity_a,
                "target": entity_b,
                "path": [],
                "length": -1
            }
        
        try:
            # Convert to undirected for shortest path
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, node_a, node_b)
            
            # Get path with edge details
            path_with_edges = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data is None:
                    edge_data = self.graph.get_edge_data(path[i+1], path[i])
                
                relation = "related_to"
                if edge_data:
                    # MultiDiGraph returns dict of dicts
                    for key, data in edge_data.items():
                        relation = data.get("relation", "related_to")
                        break
                
                path_with_edges.append({
                    "from": path[i],
                    "to": path[i+1],
                    "relation": relation
                })
            
            return {
                "found": True,
                "source": node_a,
                "target": node_b,
                "path": path,
                "path_edges": path_with_edges,
                "length": len(path) - 1
            }
        except nx.NetworkXNoPath:
            return {
                "found": False,
                "source": node_a,
                "target": node_b,
                "path": [],
                "length": -1,
                "reason": "No path exists"
            }
        except Exception as e:
            logger.warning(f"Shortest path failed: {e}")
            return {
                "found": False,
                "source": entity_a,
                "target": entity_b,
                "path": [],
                "length": -1,
                "reason": str(e)
            }
    
    def multi_hop_query(
        self,
        entities: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning across multiple entities.
        Finds common connections and paths between entities.
        
        Args:
            entities: List of entities to connect
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with multi-hop reasoning results
        """
        if len(entities) < 2:
            return {"error": "Need at least 2 entities for multi-hop query"}
        
        # Find all shortest paths between entity pairs
        connections = []
        common_entities = set()
        
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                path_result = self.get_shortest_path(e1, e2)
                if path_result["found"]:
                    connections.append({
                        "between": [e1, e2],
                        "path": path_result["path"],
                        "length": path_result["length"]
                    })
                    # Track intermediate nodes
                    if len(path_result["path"]) > 2:
                        common_entities.update(path_result["path"][1:-1])
        
        # Get BFS traversals from each entity
        traversals = {}
        for entity in entities[:3]:  # Limit to avoid explosion
            traversal = self.bfs_traverse(entity, max_depth=max_depth, max_nodes=10)
            if traversal["found"]:
                traversals[entity] = traversal["nodes"]
        
        # Find entities appearing in multiple traversals
        entity_appearances = defaultdict(int)
        for entity, nodes in traversals.items():
            for node in nodes:
                if node["entity"] != entity:
                    entity_appearances[node["entity"]] += 1
        
        hub_entities = [
            {"entity": e, "appearances": count}
            for e, count in sorted(entity_appearances.items(), key=lambda x: -x[1])
            if count > 1
        ][:5]
        
        return {
            "input_entities": entities,
            "connections": connections,
            "common_intermediate": list(common_entities),
            "hub_entities": hub_entities,
            "traversal_summaries": {
                e: len(nodes) for e, nodes in traversals.items()
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        type_counts = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            type_counts[data.get("type", "unknown")] += 1
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "entity_types": dict(type_counts),
            "most_frequent_entities": sorted(
                self.entity_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20],
        }


def main():
    """Test knowledge graph construction."""
    kg = KnowledgeGraphBuilder()
    
    # Test documents
    test_docs = [
        {
            "chunk_id": "doc1",
            "text": "Drip irrigation is highly efficient for water conservation. IoT sensors can monitor soil moisture levels in real-time, enabling precision irrigation scheduling."
        },
        {
            "chunk_id": "doc2", 
            "text": "Precision agriculture uses remote sensing and GPS technology to optimize water use efficiency. Evapotranspiration data helps determine crop water requirements."
        },
        {
            "chunk_id": "doc3",
            "text": "Soil moisture sensors like TDR probes measure volumetric water content. This data is crucial for irrigation scheduling in arid regions."
        },
    ]
    
    # Build graph
    kg.build_from_chunks(test_docs)
    
    # Print statistics
    stats = kg.get_statistics()
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Entity types: {stats['entity_types']}")
    
    # Test query
    query = "How do sensors help with drip irrigation?"
    context = kg.get_context_for_query(query)
    
    print(f"\nQuery: {query}")
    print(f"Graph context:\n{context}")


if __name__ == "__main__":
    main()
