"""
Infrastructure - Shared Resources

Singleton registry for shared resources like embedding models.
Prevents loading the same model multiple times.
"""
from typing import Optional, Dict, Any
import threading

# Lazy imports to avoid circular dependencies
_embedding_model = None
_embedding_model_lock = threading.Lock()
_llm_client = None
_llm_client_lock = threading.Lock()
_cache = None


class EmbeddingModelRegistry:
    """
    Singleton registry for embedding models.
    Ensures model is loaded only once across the application.
    """
    _instance: Optional["EmbeddingModelRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
        return cls._instance
    
    def get_model(self, model_name: str = None):
        """
        Get or create embedding model.
        
        Args:
            model_name: Model name (defaults to config)
            
        Returns:
            SentenceTransformer model instance
        """
        from sentence_transformers import SentenceTransformer
        from ..config.config import EMBEDDING_CONFIG, DEVICE
        
        if model_name is None:
            model_name = EMBEDDING_CONFIG["model_name"]
        
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    self._models[model_name] = SentenceTransformer(
                        model_name,
                        device=DEVICE
                    )
        
        return self._models[model_name]
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._models.clear()


class LLMClientRegistry:
    """
    Singleton registry for LLM clients.
    """
    _instance: Optional["LLMClientRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._clients = {}
        return cls._instance
    
    def get_client(self, client_type: str = "ollama"):
        """Get or create LLM client."""
        if client_type not in self._clients:
            with self._lock:
                if client_type not in self._clients:
                    if client_type == "ollama":
                        from langchain_community.llms import Ollama
                        from ..config.config import LLM_CONFIG
                        
                        self._clients[client_type] = Ollama(
                            model=LLM_CONFIG["model"],
                            base_url=LLM_CONFIG["base_url"],
                            temperature=LLM_CONFIG["temperature"],
                        )
                    elif client_type == "openai":
                        import openai
                        from ..config.config import LLM_CONFIG
                        
                        self._clients[client_type] = openai.OpenAI(
                            base_url=LLM_CONFIG.get("base_url", "http://localhost:11434/v1"),
                            api_key=LLM_CONFIG.get("api_key", "ollama")
                        )
        
        return self._clients.get(client_type)


class ResponseCache:
    """
    Simple in-memory cache for LLM responses.
    """
    _instance: Optional["ResponseCache"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = {}
                    cls._instance._timestamps = {}
                    cls._instance._ttl = 3600
        return cls._instance
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response."""
        import time
        
        if key in self._cache:
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp < self._ttl:
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache a response."""
        import time
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._timestamps.clear()
    
    @staticmethod
    def make_key(prompt: str, model: str = "") -> str:
        """Create cache key from prompt."""
        import hashlib
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()


# Convenience functions
def get_embedding_model(model_name: str = None):
    """Get shared embedding model instance."""
    return EmbeddingModelRegistry().get_model(model_name)


def get_llm_client(client_type: str = "ollama"):
    """Get shared LLM client instance."""
    return LLMClientRegistry().get_client(client_type)


def get_cache() -> ResponseCache:
    """Get shared response cache."""
    return ResponseCache()
