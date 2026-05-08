from fastapi import APIRouter, HTTPException, status
import logging
from pathlib import Path
from app.rag.services import (
    ChatService,
    IngestionService,
    IndexingService,
    RetrievalService,
)
from app.rag.pipeline import RAGPipeline
from app.schemas.ingestion_schema import IngestionRequest, IngestionResponse
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.schemas.indexing_schema import (
    IndexRequest,
    IndexResponse,
    DeleteIndexRequest,
    DeleteIndexResponse,
)
from app.schemas.scrape_schema import ScrapTextRequest, ScrapTextResponse
from app.rag.services.scrape_service import ScrapTextService
from app.rag.embeddings.embedder import TextEmbedder

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


# Indexing Endpoints
@router.post("/index", response_model=IndexResponse, status_code=status.HTTP_200_OK)
async def index_documents(request: IndexRequest):
    """
    Index documents and save embeddings to disk.

    - **documents**: List of structured documents to index
    """
    try:
        logger.info(f"Indexing {len(request.documents)} documents...")
        indexing_service = IndexingService()
        response = indexing_service.index_documents(
            [doc.model_dump() for doc in request.documents]
        )
        return response
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing error: {str(e)}",
        )


@router.post("/index/stop", status_code=status.HTTP_200_OK)
async def stop_indexing():
    """Stop indexing requests for the current process."""
    try:
        indexing_service = IndexingService()
        indexing_service.stop_indexing()
        return {"success": True, "message": "Indexing stop requested"}
    except Exception as e:
        logger.error(f"Error stopping indexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stop indexing error: {str(e)}",
        )


@router.delete(
    "/index", response_model=DeleteIndexResponse, status_code=status.HTTP_200_OK
)
async def delete_indexed_documents(request: DeleteIndexRequest):
    """Delete indexed documents or entire collection."""
    try:
        indexing_service = IndexingService()

        if request.delete_all:
            deleted_files = [
                str(p) for p in indexing_service.embeddings_dir.glob("*.json")
            ]
            for file_path in deleted_files:
                Path(file_path).unlink(missing_ok=True)
            indexing_service.delete_collection()
            return {
                "success": True,
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
            }

        response = indexing_service.delete_indexed_documents(request.article_ids)
        return response
    except Exception as e:
        logger.error(f"Error deleting indexed documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete indexing error: {str(e)}",
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


# Embedding Endpoint - for embedding the uploaded documents without ingesting them into the vector database
@router.post("/embed", status_code=status.HTTP_200_OK)
async def embed_text(request: IngestionRequest):
    """
    Embed text using the embedding model.

    - **documents**: List of document texts to embed

    This endpoint is for testing and debugging the embedding process without ingesting into the vector database.
    """
    try:
        logger.info(f"Embedding {len(request.documents)} documents...")
        embedder = TextEmbedder()
        embeddings = embedder.embed(request.documents)

        logger.info("Text embedding completed successfully")
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Error during embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {str(e)}",
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
async def scrape_article(request: ScrapTextRequest, save: bool = True):
    """
    Scrape text from a single article URL.

    - **url**: The URL of the article to scrape
    - **save**: Optional. If true, save the scraped article to app/data/ with unique filename

    The system will:
    1. Check if the URL is accessible
    2. Extract and clean the article HTML
    3. Extract sections from the article
    4. Optionally save to disk with unique filename (article_id_timestamp_uuid.json)
    5. Return structured article data
    """
    try:
        logger.info(f"Scraping article from URL: {request.url}")
        scrap_service = ScrapTextService()
        response = scrap_service.scrap_text(request, save_to_disk=save)

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
