from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter
import logging

from app.schemas.indexing_schema import (
    IndexResponse,
    IndexFromFetchedRequest,
    DeleteIndexRequest,
    DeleteIndexResponse,
)
from app.rag.services.indexing_service import get_indexing_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["Indexing"])


@router.post("/all", response_model=IndexResponse, status_code=status.HTTP_200_OK)
async def index_all_fetched():
    """
    Index all documents found in the fetched articles directory.
    """
    try:
        logger.info("Indexing all fetched documents...")
        indexing_service = get_indexing_service()
        response = indexing_service.index_from_fetched(article_ids=[])
        return response
    except Exception as e:
        logger.error(f"Error during index-all: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index-all error: {str(e)}",
        )


@router.post(
    "/articles", response_model=IndexResponse, status_code=status.HTTP_200_OK
)
async def index_from_fetched(request: IndexFromFetchedRequest):
    """
    Index selected fetched documents. Pass a list of article IDs to index specific documents 
    from the fetched articles directory.

    - **article_ids**: List of article IDs to index
    """
    try:
        logger.info(f"Indexing {len(request.article_ids)} fetched documents...")
        indexing_service = get_indexing_service()
        response = indexing_service.index_from_fetched(request.article_ids)
        return response
    except Exception as e:
        logger.error(f"Error during index-from-fetched: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index-from-fetched error: {str(e)}",
        )


@router.post("/stop", status_code=status.HTTP_200_OK)
async def stop_indexing():
    """Stop indexing requests for the current process."""
    try:
        indexing_service = get_indexing_service()
        indexing_service.stop_indexing()
        return {"success": True, "message": "Indexing stop requested"}
    except Exception as e:
        logger.error(f"Error stopping indexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stop indexing error: {str(e)}",
        )


@router.delete("", response_model=DeleteIndexResponse, status_code=status.HTTP_200_OK)
async def delete_indexed_documents(request: DeleteIndexRequest):
    """Delete indexed documents from Qdrant."""
    try:
        indexing_service = get_indexing_service()

        if request.delete_all or not request.article_ids:
            indexing_service.delete_collection()
            return {"success": True, "deleted_count": 0, "deleted_files": []}

        # Delete specific articles from Qdrant by article_id filter
        indexing_service.vector_db.delete_by_article_ids(
            indexing_service.collection_name, request.article_ids
        )
        return {
            "success": True,
            "deleted_count": len(request.article_ids),
            "deleted_files": [],
        }
    except Exception as e:
        logger.error(f"Error deleting indexed documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete indexing error: {str(e)}",
        )
