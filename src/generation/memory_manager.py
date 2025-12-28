"""
Conversation Memory Manager
- Maintains context across interactions
- Formats history for prompts
- Supports memory persistence
"""
from typing import List, Dict, Any, Optional

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from ..config.config import MEMORY_CONFIG, logger


class MemoryManager:
    """
    Manage conversation memory for multi-turn interactions.
    Uses LangChain's ConversationBufferWindowMemory.
    """
    
    def __init__(
        self,
        k: int = None,
        memory_key: str = None,
        return_messages: bool = None
    ):
        self.k = k or MEMORY_CONFIG["k"]
        self.memory_key = memory_key or MEMORY_CONFIG["memory_key"]
        self.return_messages = return_messages if return_messages is not None else MEMORY_CONFIG["return_messages"]
        
        self.memory = ConversationBufferWindowMemory(
            k=self.k,
            return_messages=self.return_messages,
            memory_key=self.memory_key,
            input_key="question",
            output_key="answer"
        )
        
        logger.info(f"Initialized memory manager with k={self.k}")
    
    def add_exchange(
        self,
        question: str,
        answer: str
    ):
        """
        Add a Q&A exchange to memory.
        
        Args:
            question: User question
            answer: System answer
        """
        self.memory.save_context(
            {"question": question},
            {"answer": answer}
        )
        logger.debug(f"Added exchange to memory (total: {len(self.get_messages())})")
    
    def get_context(self) -> Dict[str, Any]:
        """Get memory variables as dictionary."""
        return self.memory.load_memory_variables({})
    
    def get_messages(self) -> List[Any]:
        """Get list of messages from memory."""
        context = self.get_context()
        return context.get(self.memory_key, [])
    
    def format_for_prompt(self) -> str:
        """
        Format memory for inclusion in LLM prompt.
        
        Returns:
            Formatted conversation history string
        """
        messages = self.get_messages()
        
        if not messages:
            return ""
        
        formatted_parts = ["Previous conversation:"]
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_parts.append(f"Assistant: {msg.content}")
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "User" if msg.type == "human" else "Assistant"
                formatted_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_parts)
    
    def clear(self):
        """Clear all conversation memory."""
        self.memory.clear()
        logger.info("Cleared conversation memory")
    
    def get_summary(self) -> str:
        """Get a brief summary of the conversation."""
        messages = self.get_messages()
        num_exchanges = len(messages) // 2
        return f"Conversation has {num_exchanges} exchange(s)"
    
    def should_clear(self, new_query: str, similarity_threshold: float = 0.3) -> bool:
        """
        Determine if memory should be cleared based on topic change.
        This is a simple heuristic based on keyword overlap.
        
        Args:
            new_query: The new query
            similarity_threshold: Minimum overlap to keep memory
        
        Returns:
            True if memory should be cleared
        """
        messages = self.get_messages()
        if not messages:
            return False
        
        # Get words from previous queries
        previous_words = set()
        for msg in messages:
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                words = set(content.lower().split())
                previous_words.update(words)
        
        # Get words from new query
        new_words = set(new_query.lower().split())
        
        # Calculate overlap
        if not previous_words or not new_words:
            return False
        
        overlap = len(previous_words & new_words) / len(new_words)
        
        if overlap < similarity_threshold:
            logger.info(f"Low topic overlap ({overlap:.2f}), suggesting memory clear")
            return True
        
        return False


class ConversationState:
    """
    Extended conversation state management.
    Tracks additional context beyond raw messages.
    """
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.current_topic: Optional[str] = None
        self.sources_cited: List[str] = []
        self.entities_discussed: set = set()
        self.query_count: int = 0
    
    def process_query(
        self,
        query: str,
        response: str,
        sources: List[str] = None,
        entities: List[str] = None
    ):
        """
        Process a complete query-response cycle.
        
        Args:
            query: User query
            response: System response
            sources: Sources cited in response
            entities: Entities mentioned
        """
        # Add to memory
        self.memory_manager.add_exchange(query, response)
        
        # Track sources
        if sources:
            self.sources_cited.extend(sources)
        
        # Track entities
        if entities:
            self.entities_discussed.update(entities)
        
        self.query_count += 1
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get full conversation context."""
        return {
            "history": self.memory_manager.format_for_prompt(),
            "sources_cited": list(set(self.sources_cited[-10:])),  # Last 10 unique
            "entities_discussed": list(self.entities_discussed),
            "query_count": self.query_count,
        }
    
    def reset(self):
        """Reset conversation state."""
        self.memory_manager.clear()
        self.current_topic = None
        self.sources_cited = []
        self.entities_discussed = set()
        self.query_count = 0


def main():
    """Test memory management."""
    memory = MemoryManager()
    
    # Simulate conversation
    exchanges = [
        ("What is drip irrigation?", "Drip irrigation is a method that delivers water directly to plant roots through a network of tubes and emitters."),
        ("How efficient is it?", "Drip irrigation can achieve 90-95% water use efficiency, compared to 60-70% for sprinkler systems."),
        ("What crops benefit most?", "High-value crops like tomatoes, grapes, and fruit trees benefit most from drip irrigation due to precise water delivery."),
    ]
    
    for q, a in exchanges:
        memory.add_exchange(q, a)
    
    print("Conversation context:")
    print(memory.format_for_prompt())
    print(f"\n{memory.get_summary()}")


if __name__ == "__main__":
    main()
