import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Reranker:
    """Reranker for reordering retrieved documents based on relevance to the query."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The original query text
            documents: List of document dicts with 'combined_text' field

        Returns:
            List of document dicts sorted by relevance score (highest first)
        """
        if not documents:
            return []

        # Prepare input pairs for cross-encoder
        pairs = [[query, doc.combined_text] for doc in documents]

        # Get relevance scores from the model
        scores = self.model.predict(pairs)

        # Attach scores to documents and sort
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        reranked_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        logger.info(f"Reranked {len(documents)} documents based on relevance to the query.")

        return reranked_docs