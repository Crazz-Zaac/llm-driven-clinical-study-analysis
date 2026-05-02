import logging

from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.query_schema import QueryRequest, QueryResponse, RetrievedDocument
from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant documents from the vector database."""

    def __init__(self):
        self.embedder = TextEmbedder(show_progress=False)
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def retrieve(self, request: QueryRequest, top_k: int = 5) -> QueryResponse:
        """
        Retrieve relevant documents for a query.

        Args:
            request: QueryRequest containing the query text
            top_k: Number of top results to return

        Returns:
            QueryResponse with source documents
        """
        try:
            logger.info(f"Retrieving documents for query: {request.query[:100]}...")

            # Embed the query
            query_embedding = self.embedder.embed(request.query, show_timing=False)

            # Search for relevant documents in the vector database
            search_results = self.vector_db.search_vectors(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                top_k=top_k,
            )

            # Extract the text of the retrieved documents
            source_docs = []
            for result in search_results:
                payload = result.payload or {}
                source_docs.append(
                    RetrievedDocument(
                        article_id=payload.get("article_id", ""),
                        title=payload.get("title", ""),
                        url=payload.get("url", ""),
                        abstract=payload.get("abstract", ""),
                        score=result.score,
                    )
                )
            source_docs.sort(key=lambda x: x.score, reverse=True)  # Sort by relevance score
            top_docs = source_docs[:top_k]  # Keep only top_k documents

            logger.info(f"Retrieved {len(source_docs)} documents")

            # Return documents for the LLM to use
            return QueryResponse(response="", source_documents=top_docs)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return QueryResponse(
                response=f"Error processing query: {str(e)}", source_documents=None
            )
