from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import logging
from pathlib import Path
from typing import List

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
    IndexFromFetchedRequest,
    DeleteIndexRequest,
    DeleteIndexResponse,
)
from app.schemas.fetch_schema import (
    FetchTextBatchRequest,
    FetchTextBatchResponse,
    FetchTextListResponse,
)
from app.rag.services.fetch_service import FetchTextService
from app.rag.embeddings.embedder import TextEmbedder
from app.research.paper_fetcher import PaperFetcher
from app.schemas.paper_schema import PaperRequest, PaperResponse
from app.db import crud
from app.db.sqlite import get_db

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


@router.post(
    "/index/from-fetched", response_model=IndexResponse, status_code=status.HTTP_200_OK
)
async def index_from_fetched(request: IndexFromFetchedRequest):
    """
    Index documents by loading them from data/fetched_articles.

    - **article_ids**: List of article IDs to index
    """
    try:
        logger.info(f"Indexing {len(request.article_ids)} fetched documents...")
        indexing_service = IndexingService()
        response = indexing_service.index_from_fetched(request.article_ids)
        return response
    except Exception as e:
        logger.error(f"Error during index-from-fetched: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index-from-fetched error: {str(e)}",
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

        if request.delete_all or not request.article_ids:
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


# Article Fetching Endpoints
@router.post(
    "/fetch/from-openalex", response_model=PaperResponse, status_code=status.HTTP_200_OK
)
async def run_fetcher(request: PaperRequest, db: Session = Depends(get_db)):
    """
    Run the paper fetcher to search for articles based on a query, fetch them, and save to disk.

    - **query**: Search query to find relevant articles using OpenAlex API

    The fetcher will:
    1. Search for articles using OpenAlex API
    2. For each article, check if it's already fetched, verify with Unpaywall, check URL availability, and then fetch and save if valid.
    3. Return a summary of the fetching process.
    """
    try:
        logger.info(f"Running paper fetcher with query: {request.query}")
        fetcher = PaperFetcher()
        summary = fetcher.run(
            keywords=request.query,
            max_results=request.max_results,
            db=db,
        )
        logger.info("Paper fetcher completed successfully")
        return summary
    except Exception as e:
        logger.error(f"Error running paper fetcher: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paper fetcher error: {str(e)}",
        )


@router.post("/fetch/reset-cursor", status_code=status.HTTP_200_OK)
async def reset_cursor(query: List[str], db: Session = Depends(get_db)):
    """
    Reset the saved cursor for a given query in the paper fetcher. This will allow the fetcher to start fetching from the beginning of the results again.

    - **query**: The query for which to reset the cursor
    """
    key = " AND ".join([k.strip() for k in query])
    try:
        crud.reset_cursor(db, key)
        logger.info(f"Cursor reset successfully for query: {key}")
        return {"success": True, "message": f"Cursor reset for query: {key}"}
    except Exception as e:
        logger.error(f"Error resetting cursor for query {key}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset cursor error: {str(e)}",
        )


@router.post(
    "/fetch/batch",
    response_model=FetchTextBatchResponse,
    status_code=status.HTTP_200_OK,
)
async def fetch_articles(request: FetchTextBatchRequest, save: bool = True):
    """
    Fetch text from multiple article URLs.

    - **urls**: List of URLs to fetch
    - **save**: Optional. If true, save fetched articles to data/fetched_articles/
    """
    try:
        logger.info(f"Fetching {len(request.urls)} articles")
        fetch_service = FetchTextService()
        response = fetch_service.fetch_text_batch(request, save_to_disk=save)

        logger.info("Batch fetching completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error during batch fetching: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch fetching error: {str(e)}",
        )


@router.get(
    "/fetch/list", response_model=FetchTextListResponse, status_code=status.HTTP_200_OK
)
async def list_fetched_articles():
    """List fetched articles saved on disk."""
    try:
        fetch_service = FetchTextService()
        articles = fetch_service.list_fetched_articles()
        return {"articles": articles}
    except Exception as e:
        logger.error(f"Error listing fetched articles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List fetched articles error: {str(e)}",
        )


# Crawler Job Endpoints
@router.get("/fetch/jobs")
async def list_jobs(limit: int = 10, db: Session = Depends(get_db)):
    """List recent fetch jobs and their statuses."""
    try:
        jobs = crud.get_recent_jobs(db, limit=limit)
        return {"jobs": [job.model_dump() for job in jobs]}
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List jobs error: {str(e)}",
        )


@router.post("/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status and summary of a specific fetch job."""
    try:
        job = crud.get_job(db, job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return {"job": job.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get job status error: {str(e)}",
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
            "index": "POST /api/v1/index - Index provided documents",
            "index_from_fetched": "POST /api/v1/index/from-fetched - Index saved fetched articles",
            "retrieve": "POST /api/v1/retrieve - Retrieve documents",
            "chat": "POST /api/v1/chat - Chat with model",
            "rag": "POST /api/v1/rag - RAG pipeline (retrieve + chat)",
            "fetch_batch": "POST /api/v1/fetch/batch - Fetch article text from URLs",
            "fetch_list": "GET /api/v1/fetch/list - List fetched articles",
            "health": "GET /api/v1/health - Health check",
        },
    }
