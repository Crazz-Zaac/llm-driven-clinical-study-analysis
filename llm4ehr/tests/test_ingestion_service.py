"""Test suite for IngestionService"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag.services import IngestionService
from app.schemas.ingestion_schema import IngestionRequest


class TestIngestionService:
    """Tests for document ingestion functionality"""
    
    @pytest.fixture
    def mock_ingestion_service(self):
        """Create IngestionService with mocked dependencies"""
        with patch('app.rag.services.ingestion_service.TextChunker'), \
             patch('app.rag.services.ingestion_service.TextEmbedder'), \
             patch('app.rag.services.ingestion_service.QdrantVectorDB') as mock_db:
            service = IngestionService()
            service.vector_db = mock_db
            return service
    
    def test_ingestion_service_initialization(self, mock_ingestion_service):
        """Test IngestionService initializes correctly"""
        assert mock_ingestion_service.chunker is not None
        assert mock_ingestion_service.embedder is not None
        assert mock_ingestion_service.vector_db is not None
        assert mock_ingestion_service.collection_name is not None
    
    def test_ingest_single_document(self, mock_ingestion_service):
        """Test ingesting a single document"""
        # Mock chunk
        mock_chunk = Mock()
        mock_chunk.page_content = "This is a test chunk of clinical text."
        mock_chunk.metadata = {"source": "test.pdf", "page": 1}
        
        mock_ingestion_service.chunker.split_text = Mock(return_value=[mock_chunk])
        mock_ingestion_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_ingestion_service.vector_db.upsert_vectors = Mock()
        
        request = IngestionRequest(documents=["Test document content"])
        response = mock_ingestion_service.ingest_documents(request)
        
        assert response.success
        assert response.metadata['ingested_count'] > 0
    
    def test_ingest_multiple_documents(self, mock_ingestion_service):
        """Test ingesting multiple documents"""
        mock_chunk1 = Mock(page_content="Chunk 1", metadata={"source": "doc1.pdf", "page": 1})
        mock_chunk2 = Mock(page_content="Chunk 2", metadata={"source": "doc2.pdf", "page": 1})
        
        mock_ingestion_service.chunker.split_text = Mock(side_effect=[[mock_chunk1], [mock_chunk2]])
        mock_ingestion_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_ingestion_service.vector_db.upsert_vectors = Mock()
        
        request = IngestionRequest(documents=["Document 1", "Document 2"])
        response = mock_ingestion_service.ingest_documents(request)
        
        assert response.success
        assert mock_ingestion_service.chunker.split_text.call_count == 2
    
    def test_batch_insertion(self, mock_ingestion_service):
        """Test that vectors are inserted in batches"""
        # Create 150 mock chunks (should result in 2 batches of 100)
        mock_chunks = [
            Mock(page_content=f"Chunk {i}", metadata={"source": "test.pdf", "page": 1})
            for i in range(150)
        ]
        
        mock_ingestion_service.chunker.split_text = Mock(return_value=mock_chunks)
        mock_ingestion_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_ingestion_service.vector_db.upsert_vectors = Mock()
        
        request = IngestionRequest(documents=["Large document"])
        response = mock_ingestion_service.ingest_documents(request)
        
        # Should be called twice (100 + 50)
        assert mock_ingestion_service.vector_db.upsert_vectors.call_count == 2
        assert response.success
    
    def test_error_handling(self, mock_ingestion_service):
        """Test error handling during ingestion"""
        mock_ingestion_service.chunker.split_text = Mock(side_effect=Exception("Chunking error"))
        
        request = IngestionRequest(documents=["Test"])
        response = mock_ingestion_service.ingest_documents(request)
        
        assert not response.success
        assert "error" in str(response.metadata).lower() or "Chunking error" in str(response.metadata)
    
    def test_metadata_preservation(self, mock_ingestion_service):
        """Test that metadata is preserved during ingestion"""
        mock_chunk = Mock()
        mock_chunk.page_content = "Test content"
        mock_chunk.metadata = {"source": "clinical_paper.pdf", "page": 42}
        
        mock_ingestion_service.chunker.split_text = Mock(return_value=[mock_chunk])
        mock_ingestion_service.embedder.embed = Mock(return_value=[0.1]*384)
        mock_ingestion_service.vector_db.upsert_vectors = Mock()
        
        request = IngestionRequest(documents=["Test"])
        mock_ingestion_service.ingest_documents(request)
        
        # Check that upsert was called
        assert mock_ingestion_service.vector_db.upsert_vectors.called
        
        # Get the vectors that were passed
        call_args = mock_ingestion_service.vector_db.upsert_vectors.call_args[1]
        vectors = call_args['vectors']
        
        assert len(vectors) > 0
        assert vectors[0]['payload']['source'] == "clinical_paper.pdf"
        assert vectors[0]['payload']['page'] == 42
