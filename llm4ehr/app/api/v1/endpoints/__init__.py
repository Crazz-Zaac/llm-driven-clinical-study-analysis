"""API endpoints package"""

from .chat_routes import router as chat_router
from .fetch_routes import router as fetch_router
from .indexing_routes import router as indexing_router
from .retrieval_routes import router as retrieval_router
from .jobs_routes import router as jobs_router
from .ollama_model_routes import router as model_pull_router

__all__ = [
    "chat_router",
    "fetch_router",
    "indexing_router",
    "retrieval_router",
    "jobs_router",
    "model_pull_router",
]
