from app.rag.services import RetrievalService
from app.rag.pipeline import RAGPipeline
from app.rag.pipeline_remote import RAGPipelineRemote

_retrieval_service: RetrievalService | None = None
_rag_pipeline: RAGPipeline | None = None
_rag_pipeline_remote: RAGPipelineRemote | None = None


def get_retrieval_service() -> RetrievalService:
    """Return a process-level singleton RetrievalService.

    Using a singleton avoids hitting SQLite on every HTTP request to resolve
    the active embedding model and dimension.
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


def get_rag_pipeline() -> RAGPipeline:
    """Return a process-level singleton RAGPipeline (local Ollama model)."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def get_remote_rag_pipeline() -> RAGPipelineRemote:
    """Return a process-level singleton RAGPipelineRemote (external API model)."""
    global _rag_pipeline_remote
    if _rag_pipeline_remote is None:
        _rag_pipeline_remote = RAGPipelineRemote()
    return _rag_pipeline_remote
