"""
API Adapters Package

REST API endpoints for the RAG system.
"""
from .server import app, main as run_server

__all__ = ["app", "run_server"]
