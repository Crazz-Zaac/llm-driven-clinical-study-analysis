import logging
from collections import defaultdict
import math
import re

from app.rag.embeddings.embedder import TextEmbedder
from app.rag.reranking.reranker import Reranker
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.retrieval_schema import QueryRequest, QueryResponse, RetrievedDocument
from app.schemas.source_document_schema import SourceDocument
from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant documents from the vector database."""

    def __init__(
        self,
        max_chunks_per_article: int = 5,
        candidate_chunks_per_query: int = 200,
        reranked_chunk_limit: int = 50,
    ):
        self.embedder = TextEmbedder(show_progress=False)
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        # Controls how many top-scoring chunks are merged per article.
        # Increase for long-form section queries (e.g. methods/results),
        # decrease for lightweight deployments or smaller LLM context windows.
        self.max_chunks_per_article = max_chunks_per_article
        self.reranker = Reranker()
        self.top_k = (
            5  # Number of unique articles to return after retrieval and reranking
        )
        self.candidate_chunks_per_query = candidate_chunks_per_query
        self.reranked_chunk_limit = reranked_chunk_limit

    # this function is specifically used to clean up the section headers from the
    # combined text of the retrieved documents to prepare context for LLM evaluation
    # removes section headers like [ABSTRACT], [METHODS], [RESULTS], etc. and collapses extra blank lines
    def _strip_section_tags(self, text: str) -> str:
        text = re.sub(r"\[[A-Z\s]+\]\n?", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse extra blank lines
        return text.strip()

    def _extract_paper_title(self, query: str) -> str | None:
        """Extract quoted or 'in the paper X' style title from query."""
        # matches: "paper title" or 'paper title'
        match = re.search(r"[\"'](.*?)[\"']", query)
        if match:
            return match.group(1)
        # matches: "in the paper X, ..." or "from the paper X"
        match = re.search(
            r"(?:in|from) the paper[:\s]+(.+?)(?:,|\?|$)", query, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    def _reranker_score_to_relevance(self, score: float) -> float:
        """Map cross-encoder logits to a bounded user-facing relevance score."""
        try:
            return 1 / (1 + math.exp(-score))
        except OverflowError:
            return 0.0 if score < 0 else 1.0

    def retrieve(self, request: QueryRequest) -> QueryResponse:
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
            final_top_k = request.top_k or self.top_k

            dens_vec = self.embedder.embed_dense(request.query, show_timing=False)
            sparse_vec = self.embedder.embed_sparse(request.query)

            title_fragment = self._extract_paper_title(request.query)
            search_results = None

            # If a title fragment is found in the query, 
            # attempt to find the corresponding article ID and perform a filtered search within that article
            if title_fragment:
                logger.info(f"Extracted title fragment from query: '{title_fragment}'")
                article_id = self.vector_db.find_article_id_by_title(
                    self.collection_name, title_fragment
                )

                if article_id:
                    logger.info(
                        f"Found article ID '{article_id}' for title fragment '{title_fragment}'"
                    )
                    # Perform a filtered search within the identified article
                    # search_results = self.vector_db.search_vectors_filtered(
                    #     collection_name=self.collection_name,
                    #     query_vector=dens_vec,
                    #     article_id=article_id,
                    #     top_k=self.top_k * 10,  # fetch extra for reranking
                    # )
                    search_results = self.vector_db.hybrid_search_vectors_filtered(
                        collection_name=self.collection_name,
                        dense_vector=dens_vec,
                        sparse_vector=sparse_vec,
                        article_id=article_id,
                        top_k=max(
                            self.reranked_chunk_limit,
                            self.max_chunks_per_article * 10,
                        ),
                    )
                else:
                    logger.warning(
                        f"No article found for title fragment: '{title_fragment}', falling back to unfiltered search"
                    )
                    search_results = None

            # if no title fragment is found in the query or no article ID is found,
            # perform a standard hybrid search across all articles
            if search_results is None:
                search_results = self.vector_db.hybrid_search_vectors(
                    collection_name=self.collection_name,
                    dense_vector=dens_vec,
                    sparse_vector=sparse_vec,
                    top_k=max(self.candidate_chunks_per_query, final_top_k * 20),
                )

            logger.info("Retrieved %s chunk candidates from Qdrant", len(search_results))

            # prepare chunk candidates for reranking, filtering out any with missing text or article_id
            chunk_candidates = []
            for result in search_results:
                payload = result.payload or {}
                text = payload.get("combined_text", "")
                article_id = payload.get("article_id", "")
                if not text or not article_id:
                    continue

                chunk_candidates.append(
                    {
                        "payload": payload,
                        "score": float(result.score),
                        "retrieval_score": float(result.score),
                    }
                )

            # Rerank the chunk candidates using the cross-encoder model
            # before combining them into unique articles for the LLM to use as context
            reranked_chunks = self.reranker.rerank(
                request.query,
                chunk_candidates,
                text_getter=lambda chunk: chunk["payload"].get("combined_text", ""),
            )[: self.reranked_chunk_limit]

            logger.info(
                "Kept %s reranked chunks before article grouping", len(reranked_chunks)
            )

            # Group chunks by article
            chunks_per_article: dict = defaultdict(list)
            for chunk in reranked_chunks:
                payload = chunk["payload"]
                chunks_per_article[payload.get("article_id", "")].append(
                    {
                        "payload": payload,
                        "score": chunk["score"],
                        "retrieval_score": chunk["retrieval_score"],
                    }
                )

            # Build one RetrievedDocument per article by merging its best reranked chunks.
            # Selection is fully section-agnostic: whichever chunks score highest for the
            # query are kept, whether they come from abstract, methods, results, discussion,
            # conclusion, or any other indexed section. Merging multiple chunks prevents the
            # single highest-scoring chunk (which is often an overview-level passage) from
            # being the only content the LLM sees for that article.
            llm_docs = []
            logger.info(
                "Grouped reranked chunks into %s unique articles",
                len(chunks_per_article),
            )
            for article_id, chunks in chunks_per_article.items():
                if not chunks:
                    continue

                # 1. Pick the top reranked chunks for this article.
                top_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[
                    : self.max_chunks_per_article
                ]
                best_chunk = top_chunks[0]
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
                    logger.debug(
                        "Selected chunk article_id=%s section=%s rerank_score=%s retrieval_score=%s",
                        article_id,
                        c["payload"].get("section"),
                        c["score"],
                        c["retrieval_score"],
                    )

                merged_text = "\n\n".join(parts)
                # Preserve ordered unique section names for display
                merged_sections = ", ".join(
                    dict.fromkeys(c["payload"].get("section", "") for c in top_chunks)
                )
                ref = best_chunk["payload"]

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

            llm_docs = sorted(llm_docs, key=lambda doc: doc.score, reverse=True)[
                :final_top_k
            ]
            logger.info(
                "Returning %s unique articles after chunk reranking", len(llm_docs)
            )

            # Sort by best score and cap to the requested top_k = 5 unique articles
            # rerank the retrieved documents using the cross-encoder model
            # llm_docs = self.reranker.rerank(request.query, llm_docs)[: self.top_k]
            # print(
            #     f"Number of unique articles retrieved after reranking: {len(llm_docs)}"
            # )
            # print(
            #     f"Contents of retrieved documents after reranking: \n {[doc.combined_text for doc in llm_docs]}"
            # )

            # Converting the RetrievedDocument list to SourceDocument with score for the user response
            user_docs = [
                SourceDocument(
                    article_id=doc.article_id,
                    title=doc.title,
                    url=doc.url,
                    score=round(self._reranker_score_to_relevance(doc.score), 4),
                )
                for doc in llm_docs
            ]
            # Return documents for the LLM to use
            return QueryResponse(
                response="",
                contexts=[
                    self._strip_section_tags(doc.combined_text) for doc in llm_docs
                ],  # Provide the combined text of the retrieved documents as context only for the LLM evaluation
                llm_docs=llm_docs,
                source_documents=user_docs,
            )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return QueryResponse(response=f"Error processing query: {str(e)}")
