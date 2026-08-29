import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
import logging

from app.rag.services import RetrievalService
from app.api.dependencies.deps import get_retrieval_service
from app.schemas.retrieval_schema import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])


# Retrieval Endpoints
@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def retrieve_documents(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """
    - query: The user question.
    - retrieval_scope:
        • relevant_papers (default): Search the corpus and return the most relevant papers.
        • all_papers: Search every indexed paper individually.
        • specific_paper: Search only one paper.
    - article_id:
        Required only when retrieval_scope='specific_paper'
        and no paper title is supplied.
    - top_k:
        Number of papers to return for relevant_papers.
    """
    try:
        logger.info(f"Retrieving documents for query: {request.query[:100]}...")
        response = await asyncio.to_thread(retrieval_service.retrieve, request)
        logger.info(f"Retrieved {len(response.source_documents or [])} documents")
        return response
    except ValueError as ve:
        logger.error(f"ValueError during retrieval: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Retrieval failed due to: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}",
        )
