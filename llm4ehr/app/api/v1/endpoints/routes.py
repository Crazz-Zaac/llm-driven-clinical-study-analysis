from fastapi import APIRouter, HTTPException, status
import logging
from app.rag.services import ChatService, IngestionService, RetrievalService
from app.rag.pipeline import RAGPipeline
from app.schemas.ingestion_schema import IngestionRequest, IngestionResponse
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.schemas.scrape_schema import ScrapTextRequest, ScrapTextResponse
from app.rag.services.scrape_service import ScrapTextService

logger = logging.getLogger(__name__)

router = APIRouter()


# Ingestion Endpoints
@router.post(
    "/ingest", response_model=IngestionResponse, status_code=status.HTTP_200_OK
)
async def ingest_documents(request: IngestionRequest):
    """
    Ingest documents into the vector database.

    - **documents**: List of document texts to ingest
    - **doc_id**: Optional document ID for tracking
    """
    try:
        logger.info(f"Ingesting {len(request.documents)} documents...")
        ingestion_service = IngestionService()
        response = ingestion_service.ingest_documents(request)

        if response.success:
            logger.info(
                f"Successfully ingested {response.metadata.get('ingested_count', 0)} chunks"
            )
        else:
            logger.error(
                f"Ingestion failed: {response.metadata.get('error', 'Unknown error')}"
            )

        return response
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}",
        )


# Retrieval Endpoints
@router.post("/retrieve", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def retrieve_documents(request: QueryRequest):
    """
    Retrieve relevant documents for a query.

    - **query**: The query text to search for documents
    """
    try:
        logger.info(f"Retrieving documents for query: {request.query[:100]}...")
        retrieval_service = RetrievalService()
        response = retrieval_service.retrieve(request, top_k=5)

        logger.info(f"Retrieved {len(response.source_documents or [])} documents")
        return response
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}",
        )


# Chat Endpoints
@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """
    Chat endpoint using the chat model.

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


# Scraping Endpoint
@router.post(
    "/scrape", response_model=ScrapTextResponse, status_code=status.HTTP_200_OK
)
async def scrape_article(request: ScrapTextRequest):
    """
    Scrape text from a single article URL.

    - **url**: The URL of the article to scrape

    The system will:
    1. Check if the URL is accessible
    2. Extract and clean the article HTML
    3. Extract sections from the article
    4. Return structured article data
    """
    try:
        logger.info(f"Scraping article from URL: {request.url}")
        scrap_service = ScrapTextService()
        response = scrap_service.scrap_text(request)

        logger.info("Article scraped successfully")
        return response
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping error: {str(e)}",
        )


# Information Endpoints
@router.get("/info")
async def get_info():
    """Get information about the API"""
    return {
        "service": "Clinical Study Analysis RAG",
        "version": "0.1.0",
        "endpoints": {
            "ingest": "POST /api/v1/ingest - Ingest documents",
            "retrieve": "POST /api/v1/retrieve - Retrieve documents",
            "chat": "POST /api/v1/chat - Chat with model",
            "rag": "POST /api/v1/rag - RAG pipeline (retrieve + chat)",
            "scrape": "POST /api/v1/scrape - Scrape article text from URL",
            "health": "GET /api/v1/health - Health check",
        },
    }
