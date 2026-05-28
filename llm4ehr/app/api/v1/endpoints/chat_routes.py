from fastapi import APIRouter, HTTPException, logger, status
from fastapi import APIRouter
import logging

from app.rag.pipeline import RAGPipeline
from app.rag.pipeline_remote import RAGPipelineRemote
from app.schemas.chat_schema import ChatRequest, ChatResponse

router = APIRouter(prefix="", tags=["Chat"])
logger = logging.getLogger(__name__)


# Local RAG pipeline endpoint
@router.post("/rag/local", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def rag_chat(request: ChatRequest):
    """
    RAG pipeline endpoint: retrieves documents and generates response using chat model.

    - **messages**: List of chat messages. Last user message will be used as query.

    The system will:
    1. Extract user query from messages
    2. Retrieve relevant documents
    3. Generate response using those documents as context
    """
    try:
        logger.info("Processing RAG pipeline request...")
        rag_pipeline = RAGPipeline()
        response = rag_pipeline.run(request)

        if response.response:
            logger.info("RAG response generated successfully")
        else:
            logger.warning("RAG pipeline returned empty response")

        return response
    except Exception as e:
        logger.error(f"Error during RAG pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {str(e)}",
        )


# REmote RAG pipeline endpoint (calls external API)
@router.post("/rag/remote", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def rag_chat_remote(request: ChatRequest):
    """
    RAG pipeline endpoint that calls an external API for processing.

    This is useful for testing the RAG pipeline without relying on the local Ollama model.
    The external API should implement the same logic as the local RAGPipeline but can use a different LLM.
    """
    try:
        logger.info("Processing remote RAG pipeline request...")
        rag_pipeline = RAGPipelineRemote()
        response = rag_pipeline.run(request)

        if response.response:
            logger.info("Remote RAG response generated successfully")
        else:
            logger.warning("Remote RAG pipeline returned empty response")

        return response
    except Exception as e:
        logger.error(f"Error during remote RAG pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Remote RAG pipeline error: {str(e)}",
        )
