"""Test suite for RetrievalService."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.rag.services import RetrievalService
from app.schemas.retrieval_schema import QueryRequest


def make_result(
    article_id: str,
    text: str,
    score: float,
    chunk_index: int = 0,
    title: str | None = None,
):
    return SimpleNamespace(
        score=score,
        payload={
            "article_id": article_id,
            "title": title or f"Paper {article_id}",
            "url": f"https://example.test/{article_id}",
            "abstract": f"Abstract {article_id}",
            "combined_text": text,
            "section": "results",
            "chunk_index": chunk_index,
        },
    )


class TestRetrievalService:
    """Tests for query/retrieval functionality."""

    @pytest.fixture
    def mock_retrieval_service(self):
        """Create RetrievalService with mocked dependencies."""
        with patch("app.rag.services.retrieval_service.TextEmbedder"), patch(
            "app.rag.services.retrieval_service.QdrantVectorDB"
        ), patch("app.rag.services.retrieval_service.Reranker"):
            service = RetrievalService(
                max_chunks_per_article=2,
                candidate_chunks_per_query=4,
                reranked_chunk_limit=3,
            )
            service.embedder.embed_dense = Mock(return_value=[0.1] * 384)
            service.embedder.embed_sparse = Mock(
                return_value={"indices": [1, 2], "values": [0.7, 0.3]}
            )
            service.reranker.rerank.side_effect = (
                lambda query, chunks, text_getter=None: sorted(
                    chunks, key=lambda chunk: chunk["score"], reverse=True
                )
            )
            return service

    def test_retrieval_service_initialization(self, mock_retrieval_service):
        """Test RetrievalService initializes correctly."""
        assert mock_retrieval_service.collection_name is not None
        assert mock_retrieval_service.embedder is not None
        assert mock_retrieval_service.vector_db is not None
        assert mock_retrieval_service.reranker is not None

    def test_retrieve_success_groups_reranked_chunks_by_article(
        self, mock_retrieval_service
    ):
        """Test successful document retrieval after chunk-level reranking."""
        mock_retrieval_service.vector_db.hybrid_search_vectors.return_value = [
            make_result("A", "second relevant chunk", 0.9, chunk_index=2),
            make_result("B", "another paper chunk", 0.8, chunk_index=1),
            make_result("A", "first relevant chunk", 0.7, chunk_index=1),
        ]

        response = mock_retrieval_service.retrieve(QueryRequest(query="clinical result"))

        assert [doc.article_id for doc in response.llm_docs] == ["A", "B"]
        assert response.llm_docs[0].combined_text.index("first relevant") < response.llm_docs[
            0
        ].combined_text.index("second relevant")
        assert response.source_documents[0].article_id == "A"

    def test_chunk_reranking_can_rescue_lower_retrieval_score_article(
        self, mock_retrieval_service
    ):
        """The reranker should run before article grouping and final top-k selection."""

        def rerank_by_target_text(query, chunks, text_getter=None):
            for chunk in chunks:
                text = text_getter(chunk) if text_getter else ""
                chunk["score"] = 10.0 if "target evidence" in text else 0.1
            return sorted(chunks, key=lambda chunk: chunk["score"], reverse=True)

        mock_retrieval_service.reranker.rerank.side_effect = rerank_by_target_text
        mock_retrieval_service.vector_db.hybrid_search_vectors.return_value = [
            make_result("A", "generic high retrieval chunk", 0.99),
            make_result("A", "another generic high retrieval chunk", 0.98),
            make_result("B", "target evidence from another paper", 0.2),
        ]

        response = mock_retrieval_service.retrieve(
            QueryRequest(query="find target evidence", top_k=1)
        )

        assert [doc.article_id for doc in response.llm_docs] == ["B"]

    def test_retrieve_no_results(self, mock_retrieval_service):
        """Test retrieval when no documents are found."""
        mock_retrieval_service.vector_db.hybrid_search_vectors.return_value = []

        response = mock_retrieval_service.retrieve(QueryRequest(query="missing topic"))

        assert response.llm_docs == []
        assert response.source_documents == []

    def test_retrieve_error_handling(self, mock_retrieval_service):
        """Test error handling in retrieval."""
        mock_retrieval_service.embedder.embed_dense.side_effect = Exception(
            "Embedding error"
        )

        response = mock_retrieval_service.retrieve(QueryRequest(query="Test query"))

        assert "Error" in response.response
        assert response.source_documents == []

    def test_embedding_called_with_query(self, mock_retrieval_service):
        """Test that dense and sparse embedding are called with the query text."""
        mock_retrieval_service.vector_db.hybrid_search_vectors.return_value = []
        query_text = "What is the latest treatment?"

        mock_retrieval_service.retrieve(QueryRequest(query=query_text))

        mock_retrieval_service.embedder.embed_dense.assert_called_once_with(
            query_text, show_timing=False
        )
        mock_retrieval_service.embedder.embed_sparse.assert_called_once_with(query_text)

    def test_retrieve_with_custom_top_k(self, mock_retrieval_service):
        """Test retrieval respects QueryRequest top_k for final article count."""
        mock_retrieval_service.vector_db.hybrid_search_vectors.return_value = [
            make_result("A", "chunk A", 0.9),
            make_result("B", "chunk B", 0.8),
            make_result("C", "chunk C", 0.7),
        ]

        response = mock_retrieval_service.retrieve(QueryRequest(query="Test", top_k=2))

        assert [doc.article_id for doc in response.llm_docs] == ["A", "B"]
        call_args = mock_retrieval_service.vector_db.hybrid_search_vectors.call_args
        assert call_args.kwargs["collection_name"] == mock_retrieval_service.collection_name
