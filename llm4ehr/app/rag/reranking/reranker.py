import logging
from collections.abc import Callable

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker for reordering retrieved documents based on relevance to the query."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list,
        text_getter: Callable | None = None,
        score_setter: Callable | None = None,
    ) -> list:
        """
        Rerank documents or chunk candidates based on relevance to the query.

        Args:
            query: The original query text
            documents: Items to rerank
            text_getter: Optional callable for extracting rerankable text
            score_setter: Optional callable for writing the reranker score

        Returns:
            List of items sorted by relevance score (highest first)
        """
        if not documents:
            return []

        # helpers for getting text and setting score, with defaults if not provided
        def default_text_getter(doc) -> str:
            if isinstance(doc, dict):
                return doc.get("combined_text") or doc.get("text") or ""
            return getattr(doc, "combined_text", "")

        # helper for setting score, with default if not provided
        def default_score_setter(doc, score: float) -> None:
            if isinstance(doc, dict):
                doc["score"] = score
            else:
                doc.score = score

        get_text = text_getter or default_text_getter
        set_score = score_setter or default_score_setter

        # prepare pairs of (query, document text) for the cross-encoder model
        pairs = [[query, get_text(doc)] for doc in documents]

        # predict the scores for each document with respect to the query using the cross-encoder model
        scores = self.model.predict(pairs)

        # set the scores on the documents 
        for doc, score in zip(documents, scores):
            set_score(doc, float(score))

        # sort the documents by score in descending order
        reranked_docs = sorted(
            documents,
            key=lambda doc: doc["score"] if isinstance(doc, dict) else doc.score,
            reverse=True,
        )
        logger.info(
            "Reranked %s candidates based on relevance to the query.", len(documents)
        )

        return reranked_docs
