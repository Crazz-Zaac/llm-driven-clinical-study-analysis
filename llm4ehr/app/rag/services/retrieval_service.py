import logging
from collections import defaultdict

from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.retrieval_schema import QueryRequest, QueryResponse, RetrievedDocument
from app.schemas.source_document_schema import SourceDocument
from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant documents from the vector database."""

    def __init__(self, max_chunks_per_article: int = 3):
        self.embedder = TextEmbedder(show_progress=False)
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        # Controls how many top-scoring chunks are merged per article.
        # Increase for long-form section queries (e.g. methods/results),
        # decrease for lightweight deployments or smaller LLM context windows.
        self.max_chunks_per_article = max_chunks_per_article

    def retrieve(self, request: QueryRequest, top_k: int = 5, min_score: float = 0.0) -> QueryResponse:
        """
        Retrieve relevant documents for a query.

        Args:
            request: QueryRequest containing the query text
            top_k: Number of unique articles to return
            min_score: Minimum cosine similarity score to accept (0.0 = no filter)

        Returns:
            QueryResponse with source documents
        """
        try:
            logger.info(f"Retrieving documents for query: {request.query[:100]}...")

            # Embed the query
            query_embedding = self.embedder.embed(request.query, show_timing=False)

            # Fetch extra candidates so per-article deduplication still yields top_k articles
            search_results = self.vector_db.search_vectors(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                top_k=top_k * 3,
            )

            # Apply minimum similarity score filter
            if min_score > 0.0:
                search_results = [r for r in search_results if r.score >= min_score]

            # Group chunks by article
            chunks_per_article: dict = defaultdict(list)
            for result in search_results:
                payload = result.payload or {}
                chunks_per_article[payload.get("article_id", "")].append(
                    {"payload": payload, "score": result.score}
                )

            # Build one RetrievedDocument per article by merging its top-scoring chunks.
            # Selection is fully section-agnostic: whichever chunks score highest for the
            # query are kept, whether they come from abstract, methods, results, discussion,
            # conclusion, or any other indexed section. Merging multiple chunks prevents the
            # single highest-scoring chunk (which is often an overview-level passage) from
            # being the only content the LLM sees for that article.
            llm_docs = []
            print(f"Length of chunks_per_article: {len(chunks_per_article)}")
            for article_id, chunks in chunks_per_article.items():
                if not chunks:
                    continue

                # 1. Pick the top-scoring chunks
                top_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[:self.max_chunks_per_article]
                best_score = top_chunks[0]["score"]

                # 2. Re-sort by chunk_index so merged text reads in document order
                top_chunks.sort(key=lambda x: x["payload"].get("chunk_index", 0))

                # 3. Concatenate text with section headers
                parts = []
                for c in top_chunks:
                    section = c["payload"].get("section", "")
                    text = c["payload"].get("combined_text", "")
                    if text:
                        parts.append(f"[{section.upper()}]\n{text}")

                merged_text = "\n\n".join(parts)
                # Preserve ordered unique section names for display
                merged_sections = ", ".join(
                    dict.fromkeys(c["payload"].get("section", "") for c in top_chunks)
                )
                ref = top_chunks[0]["payload"]

                llm_docs.append(
                    RetrievedDocument(
                        article_id=ref.get("article_id", ""),
                        title=ref.get("title", ""),
                        url=ref.get("url", ""),
                        abstract=ref.get("abstract", ""),
                        combined_text=merged_text,
                        section=merged_sections,
                        score=best_score,
                    )
                )

            # Sort by best score and cap to the requested top_k unique articles
            llm_docs = sorted(llm_docs, key=lambda d: d.score, reverse=True)[:top_k]
            print(f"Contents of retrieved document: \n {[doc.combined_text[:200] for doc in llm_docs]}")
            print(f"")

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
            return QueryResponse(response=f"Error processing query: {str(e)}")
