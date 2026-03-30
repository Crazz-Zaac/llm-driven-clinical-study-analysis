"""Test suite for RetrievalService (retrieval)"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from app.rag.services import RetrievalService
from app.schemas.query_schema import QueryRequest


class TestRetrievalService:
    """Tests for query/retrieval functionality"""
    
    @pytest.fixture
    def mock_retrieval_service(self):
        """Create RetrievalService with mocked dependencies"""
        with patch('app.rag.services.retrieval_service.TextEmbedder'), \
             patch('app.rag.services.retrieval_service.QdrantVectorDB') as mock_db:
            service = RetrievalService()
            service.vector_db = mock_db
            return service
    
    def test_retrieval_service_initialization(self, mock_retrieval_service):
        """Test RetrievalService initializes correctly"""
        assert mock_retrieval_service.collection_name is not None
        assert mock_retrieval_service.embedder is not None
        assert mock_retrieval_service.vector_db is not None
    
    def test_retrieve_success(self, mock_retrieval_service):
        """Test successful document retrieval"""
        # Mock embedding
        mock_retrieval_service.embedder.embed = Mock(return_value=[0.1]*384)
        
        # Mock search results
        mock_result = Mock()
        mock_result.payload = {"text": "Sample clinical document about disease X"}
        mock_retrieval_service.vector_db.search_vectors = Mock(return_value=[mock_result])
        
        # Retrieve documents
        request = QueryRequest(query="What is disease X?")
        response = mock_retrieval_service.retrieve(request)
        
        assert response.source_documents is not None
        assert len(response.source_documents) > 0
        assert "disease X" in response.source_documents[0]
    
    def test_retrieve_no_results(self, mock_retrieval_service):
        """Test retrieval when no documents found"""
        mock_retrieval_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_retrieval_service.vector_db.search_vectors = Mock(return_value=[])
        
        request = QueryRequest(query="Non-existent topic")
        response = mock_retrieval_service.retrieve(request)
        
        assert response.source_documents == []
    
    def test_retrieve_error_handling(self, mock_retrieval_service):
        """Test error handling in retrieval"""
        mock_retrieval_service.embedder.embed = Mock(side_effect=Exception("Embedding error"))
        
        request = QueryRequest(query="Test query")
        response = mock_retrieval_service.retrieve(request)
        
        assert "Error" in response.response
        assert response.source_documents is None
    
    def test_embedding_called_with_query(self, mock_retrieval_service):
        """Test that embedding is called with the query text"""
        mock_retrieval_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_retrieval_service.vector_db.search_vectors = Mock(return_value=[])
        
        query_text = "What is the latest treatment?"
        request = QueryRequest(query=query_text)
        mock_retrieval_service.retrieve(request)
        
        mock_retrieval_service.embedder.embed.assert_called_once()
    
    def test_retrieve_with_custom_top_k(self, mock_retrieval_service):
        """Test retrieval with custom top_k parameter"""
        mock_retrieval_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_retrieval_service.vector_db.search_vectors = Mock(return_value=[])
        
        request = QueryRequest(query="Test")
        mock_retrieval_service.retrieve(request, top_k=10)
        
        mock_retrieval_service.vector_db.search_vectors.assert_called_once()
        call_args = mock_retrieval_service.vector_db.search_vectors.call_args
        
        assert call_args[1]['collection_name'] == mock_retrieval_service.collection_name
        assert call_args[1]['top_k'] == 10
