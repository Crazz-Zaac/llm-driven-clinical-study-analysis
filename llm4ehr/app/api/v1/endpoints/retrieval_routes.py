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
    Retrieve relevant documents for a query.

    - **query**: The query text to search for documents
    - **top_k**: Number of documents to retrieve (1–20, default 5)
    """
    try:
        logger.info(f"Retrieving documents for query: {request.query[:100]}...")
        response = retrieval_service.retrieve(request, top_k=request.top_k)

        logger.info(f"Retrieved {len(response.source_documents or [])} documents")
        return response
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}",
        )
