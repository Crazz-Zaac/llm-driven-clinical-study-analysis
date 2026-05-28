import logging

from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.retrieval_schema import QueryRequest, QueryResponse, RetrievedDocument
from app.schemas.source_document_schema import SourceDocument
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

            # For LLM context: merge chunks per article to provide more coherent information
            from collections import defaultdict

            chunks_per_article = defaultdict(list)
            for result in search_results:
                payload = result.payload or {}
                chunks_per_article[payload.get("article_id", "")].append(
                    {
                        "payload": payload,
                        "score": result.score,
                    }
                )

            # Extract the text of the retrieved documents
            llm_docs = []
            for article_id, chunks in chunks_per_article.items():
                if not chunks:
                    continue
                # Sort chunks by score and take the top one for the article
                best_chunk = max(chunks, key=lambda x: x["score"])
                payload = best_chunk["payload"]
                llm_docs.append(
                    RetrievedDocument(
                        article_id=payload.get("article_id", ""),
                        title=payload.get("title", ""),
                        url=payload.get("url", ""),
                        abstract=payload.get("abstract", ""),
                        combined_text=payload.get("combined_text", ""),
                        section=payload.get("section", ""),
                        score=best_chunk["score"],
                    )
                )

            # Converting the RetrievedDocument list to SourceDocument with score for the user response
            user_docs = [
                SourceDocument(
                    article_id=doc.article_id,
                    title=doc.title,
                    url=doc.url,
                    score=round(doc.score, 4),
                )
                for doc in llm_docs
            ]
            # Return documents for the LLM to use
            return QueryResponse(
                response="", llm_docs=llm_docs, source_documents=user_docs
            )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return QueryResponse(
                response=f"Error processing query: {str(e)}", source_documents=None
            )
