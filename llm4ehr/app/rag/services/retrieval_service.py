import logging
from collections import defaultdict
import math
import re

from app.rag.embeddings.embedder import TextEmbedder
from app.rag.reranking.reranker import Reranker
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.retrieval_schema import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    RetrievalScope,
)
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
        self.max_chunks_per_relevant_article = 5
        self.max_chunks_per_all_papers_article = 2

    # this function is specifically used to clean up the section headers from the
    # combined text of the retrieved documents to prepare context for LLM evaluation
    # removes section headers like [ABSTRACT], [METHODS], [RESULTS], etc. and collapses extra blank lines
    def _strip_section_tags(self, text: str) -> str:
        text = re.sub(r"\[[A-Z\s]+\]\n?", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse extra blank lines
        return text.strip()

    def _extract_paper_title(self, query: str) -> str | None:
        """Extract quoted or paper-reference style title from query."""
        # matches: "paper title" or 'paper title'
        match = re.search(r"[\"'](.*?)[\"']", query)
        if match:
            return match.group(1)
        # matches: "in/from/according to the paper X" with optional colon
        match = re.search(
            r"(?:in|from|according to) the paper[:\s]+(.+?)(?:,|\?|$)",
            query,
            re.IGNORECASE,
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

    def _prepare_chunk_candidates(self, search_results: list[dict]) -> list[dict]:
        """Prepare chunk candidates for reranking, filtering out any with missing text or article_id."""
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
        return chunk_candidates


    def _build_response_from_grouped_chunks(
        self,
        request: QueryRequest,
        chunks_per_article: dict[str, list[dict]],
        limit_articles: bool = True,
        max_chunks_per_article: int | None = None,
    ) -> QueryResponse:
        """
        Build one RetrievedDocument per article by merging its best reranked chunks.
        Selection is fully section-agnostic: whichever chunks score highest for the
        query are kept, whether they come from abstract, methods, results, discussion,
        conclusion, or any other indexed section. Merging multiple chunks prevents the
        single highest-scoring chunk (which is often an overview-level passage) from
        being the only content the LLM sees for that article.
        """

        chunk_limit = (
            max_chunks_per_article
            if max_chunks_per_article is not None
            else self.max_chunks_per_article
        )

        llm_docs = []
        for article_id, chunks in chunks_per_article.items():
            if not chunks:
                continue

            # 1. Pick the top reranked chunks for this article.
            top_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[
                :chunk_limit
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
                logger.info(
                    "Selected article=%s section=%s score=%.4f text=%r",
                    article_id,
                    c["payload"].get("section"),
                    c["score"],
                    c["payload"].get("combined_text", "")[:300],
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

        llm_docs = sorted(llm_docs, key=lambda doc: doc.score, reverse=True)
        logger.info(
            "Returning articles: %s",
            [
                {
                    "article_id": doc.article_id,
                    "title": doc.title,
                    "url": doc.url,
                    "score": doc.score,
                }
                for doc in llm_docs
            ],
        )

        # true for _retrieve_across_all_papers and _retrieve_relevant_papers,
        # false for _retrieve_from_specific_paper
        if limit_articles:
            final_top_k = request.top_k or self.top_k
            llm_docs = llm_docs[:final_top_k]

        # logger.info("Returning %s unique articles after chunk reranking", len(llm_docs))

        return QueryResponse(llm_docs=llm_docs)

    def _retrieve_across_all_papers(
        self,
        request: QueryRequest,
        dense_vector: list[float],
        sparse_vector: dict,
    ) -> QueryResponse:
        """Perform retrieval across all papers in the corpus."""

        # get all the article IDs in the collection
        all_article_ids = self.vector_db.get_all_article_ids(self.collection_name)

        candidates_by_article: dict[str, list[dict]] = defaultdict(list)

        for article_id in all_article_ids:
            # Perform a hybrid search for each article
            search_results = self.vector_db.hybrid_search_vectors_filtered(
                collection_name=self.collection_name,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                article_id=article_id,
                top_k=max(
                    self.reranked_chunk_limit,
                    self.max_chunks_per_article * 10,
                ),
            )

            chunk_candidates = self._prepare_chunk_candidates(search_results)

            # Rerank the chunk candidates using the cross-encoder model
            # before combining them into unique articles for the LLM to use as context
            reranked_chunks = self.reranker.rerank(
                request.query,
                chunk_candidates,
                text_getter=lambda chunk: chunk["payload"].get("combined_text", ""),
            )  # [: self.reranked_chunk_limit]

            candidates_by_article[article_id] = reranked_chunks[
                : self.reranked_chunk_limit
            ]

        return self._build_response_from_grouped_chunks(
            request,
            candidates_by_article,
            limit_articles=False,
            max_chunks_per_article=2,
        )

    def _retrieve_from_specific_paper(
        self,
        request: QueryRequest,
        dense_vector: list[float],
        sparse_vector: dict,
    ) -> QueryResponse:
        """Perform retrieval from a specific paper based on the provided article_id."""
        if not request.article_id:
            raise ValueError(
                "article_id must be provided when retrieval_scope is 'specific_paper'."
            )

        # Perform a hybrid search for the specific article
        # This search is filtered to only include chunks from the specified article_id.
        search_results = self.vector_db.hybrid_search_vectors_filtered(
            collection_name=self.collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            article_id=request.article_id,
            top_k=max(
                self.reranked_chunk_limit,
                self.max_chunks_per_article * 10,
            ),
        )

        chunk_candidates = self._prepare_chunk_candidates(search_results)

        # Rerank the chunk candidates using the cross-encoder model
        # before combining them into unique articles for the LLM to use as context
        reranked_chunks = self.reranker.rerank(
            request.query,
            chunk_candidates,
            text_getter=lambda chunk: chunk["payload"].get("combined_text", ""),
        )  # [: self.reranked_chunk_limit]

        return self._build_response_from_grouped_chunks(
            request, {request.article_id: reranked_chunks[: self.reranked_chunk_limit]}
        )

    def _retrieve_relevant_papers(
        self,
        request: QueryRequest,
        dense_vector: list[float],
        sparse_vector: dict,
    ) -> QueryResponse:
        """Single corpus-wide hybrid search (no per-article filter), grouped into articles."""
        search_results = self.vector_db.hybrid_search_vectors(
            collection_name=self.collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=self.candidate_chunks_per_query,
        )

        chunk_candidates = self._prepare_chunk_candidates(search_results)

        reranked_chunks = self.reranker.rerank(
            request.query,
            chunk_candidates,
            text_getter=lambda chunk: chunk["payload"].get("combined_text", ""),
        )[: self.reranked_chunk_limit]

        chunks_per_article: dict[str, list[dict]] = defaultdict(list)
        for chunk in reranked_chunks:
            article_id = chunk["payload"].get("article_id", "")
            if article_id:
                chunks_per_article[article_id].append(chunk)

        return self._build_response_from_grouped_chunks(request, chunks_per_article)

    def _resolve_specific_paper_request(self, request: QueryRequest) -> QueryRequest:
        """Ensure a specific-paper request has an article_id, resolving it from the query title when needed."""
        if request.article_id:
            return request

        title_fragment = self._extract_paper_title(request.query)
        if not title_fragment:
            raise ValueError(
                "specific_paper retrieval requires article_id or a paper title in the query."
            )

        logger.info(
            "Looking up article ID for specific-paper title fragment: %s",
            title_fragment,
        )
        article_id = self.vector_db.find_article_id_by_title(
            self.collection_name, title_fragment
        )
        if not article_id:
            raise ValueError(
                f"No indexed article found for paper title fragment: {title_fragment}"
            )

        return request.model_copy(update={"article_id": article_id})

    def retrieve(self, request: QueryRequest) -> QueryResponse:
        """
        Retrieve relevant documents for a query.

        Args:
            request: QueryRequest containing the query text
        Returns:
            QueryResponse with source documents
        """
        try:
            logger.info(f"Retrieving documents for query: {request.query[:100]}...")

            dens_vec = self.embedder.embed_dense(request.query, show_timing=False)
            sparse_vec = self.embedder.embed_sparse(request.query)

            if request.retrieval_scope == RetrievalScope.SPECIFIC_PAPER:
                resolved_request = self._resolve_specific_paper_request(request)
                logger.info(
                    "Retrieving from specific paper with article_id: %s",
                    resolved_request.article_id,
                )
                retrieved_response = self._retrieve_from_specific_paper(

                    resolved_request, dens_vec, sparse_vec
                )
            elif request.retrieval_scope == RetrievalScope.RELEVANT_PAPERS:
                logger.info("Retrieving relevant papers based on query")
                retrieved_response = self._retrieve_relevant_papers(
                    request, dens_vec, sparse_vec
                )
            elif request.retrieval_scope == RetrievalScope.ALL_PAPERS:
                logger.info("Retrieving across all papers in the corpus")
                retrieved_response = self._retrieve_across_all_papers(
                    request, dens_vec, sparse_vec
                )
            else:
                raise ValueError(
                    f"Invalid retrieval_scope: {request.retrieval_scope}. Must be one of: {list(RetrievalScope)}"
                )

            user_docs = [
                SourceDocument(
                    article_id=doc.article_id,
                    title=doc.title,
                    url=doc.url,
                    score=round(self._reranker_score_to_relevance(doc.score), 4),
                )
                for doc in retrieved_response.llm_docs
            ]

            # label the context
            contexts = [
                (
                    f"ARTICLE ID: {doc.article_id}\n"
                    f"TITLE: {doc.title}\n"
                    f"SECTIONS: {doc.section}\n\n"
                    f"{self._strip_section_tags(doc.combined_text)}"
                )
                for doc in retrieved_response.llm_docs
            ]

            # Return documents for the LLM to use
            return QueryResponse(
                response="",
                contexts=contexts,  # Provide the combined text of the retrieved documents as context only for the LLM evaluation
                llm_docs=retrieved_response.llm_docs,
                source_documents=user_docs,
            )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise ValueError(f"Retrieval failed: {str(e)}")
