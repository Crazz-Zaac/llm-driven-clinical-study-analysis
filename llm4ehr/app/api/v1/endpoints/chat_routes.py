from fastapi import APIRouter, HTTPException, logger, status
from fastapi import APIRouter
import logging
from app.rag.services import (
    ChatService,
)
from app.rag.pipeline import RAGPipeline
from app.schemas.chat_schema import ChatRequest, ChatResponse

router = APIRouter(prefix="", tags=["Chat"])
logger = logging.getLogger(__name__)

# Chat Endpoints
@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """
    Chat endpoint to chat with the model but without retrieval. Useful for testing the chat model directly.

    - **messages**: List of chat messages with role and content
    """
    try:
        logger.info("Processing chat request...")
        chat_service = ChatService()
        response = chat_service.handle_chat(request)

        logger.info("Chat response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )


# RAG Pipeline Endpoints
@router.post("/rag", response_model=ChatResponse, status_code=status.HTTP_200_OK)
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
