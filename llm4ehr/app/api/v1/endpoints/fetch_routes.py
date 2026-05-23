from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import logging
from typing import List

from app.schemas.fetch_schema import (
    FetchTextBatchRequest,
    FetchTextBatchResponse,
    FetchTextListResponse,
)
from app.rag.services.fetch_service import FetchTextService
from app.research.paper_fetcher import PaperFetcher
from app.schemas.paper_schema import PaperRequest, PaperResponse
from app.db import crud
from app.db.sqlite import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fetch", tags=["Fetching-paper"])


# Article Fetching Endpoints
@router.post(
    "/from-openalex", response_model=PaperResponse, status_code=status.HTTP_200_OK
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


@router.post("/reset-cursor", status_code=status.HTTP_200_OK)
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
    "/batch",
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
    "/list", response_model=FetchTextListResponse, status_code=status.HTTP_200_OK
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
