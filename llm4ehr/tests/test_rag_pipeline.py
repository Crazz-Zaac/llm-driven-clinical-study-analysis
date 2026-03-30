"""Test suite for RAGPipeline"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag.pipeline import RAGPipeline
from app.schemas.chat_schema import ChatRequest, ChatMessage


class TestRAGPipeline:
    """Tests for RAG pipeline end-to-end functionality"""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create RAGPipeline with mocked dependencies"""
        with patch('app.rag.pipeline.RetrievalService'), \
             patch('app.rag.pipeline.ChatModel') as mock_chat:
            pipeline = RAGPipeline()
            return pipeline
    
    def test_pipeline_initialization(self, mock_rag_pipeline):
        """Test RAGPipeline initializes correctly"""
        assert mock_rag_pipeline.retrieval_service is not None
        assert mock_rag_pipeline.chat_model is not None
    
    def test_pipeline_run_with_documents(self, mock_rag_pipeline):
        """Test RAG pipeline with documents found"""
        # Mock retrieval
        mock_retrieval_response = Mock()
        mock_retrieval_response.source_documents = ["Document about disease X", "Another document"]
        mock_rag_pipeline.retrieval_service.retrieve = Mock(return_value=mock_retrieval_response)
        
        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.response = "Disease X is..."
        mock_rag_pipeline.chat_model.generate_response = Mock(return_value=mock_chat_response)
        
        # Create request
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="What is disease X?")
        ])
        
        # Run pipeline
        response = mock_rag_pipeline.run(request)
        
        # Verify retrieval was called
        assert mock_rag_pipeline.retrieval_service.retrieve.called
        
        # Verify chat model was called with system prompt containing documents
        assert mock_rag_pipeline.chat_model.generate_response.called
        call_args = mock_rag_pipeline.chat_model.generate_response.call_args[0]
        chat_request = call_args[0]
        
        # System message should contain retrieved documents
        system_msg = [m for m in chat_request.messages if m.role == "system"][0]
        assert "disease X" in system_msg.content.lower() or "Document" in system_msg.content
    
    def test_pipeline_run_without_documents(self, mock_rag_pipeline):
        """Test RAG pipeline when no documents are found"""
        # Mock retrieval with no results
        mock_retrieval_response = Mock()
        mock_retrieval_response.source_documents = []
        mock_rag_pipeline.retrieval_service.retrieve = Mock(return_value=mock_retrieval_response)
        
        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.response = "I couldn't find relevant documents"
        mock_rag_pipeline.chat_model.generate_response = Mock(return_value=mock_chat_response)
        
        # Create request
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="What is unknown disease?")
        ])
        
        # Run pipeline
        response = mock_rag_pipeline.run(request)
        
        # Verify chat model was called
        assert mock_rag_pipeline.chat_model.generate_response.called
        
        # System message should indicate no documents found
        call_args = mock_rag_pipeline.chat_model.generate_response.call_args[0]
        chat_request = call_args[0]
        system_msg = [m for m in chat_request.messages if m.role == "system"][0]
        assert "No relevant documents" in system_msg.content
    
    def test_pipeline_no_user_message(self, mock_rag_pipeline):
        """Test RAG pipeline with no user messages"""
        # Create request with only system message
        request = ChatRequest(messages=[
            ChatMessage(role="system", content="You are a helpful assistant")
        ])
        
        # Run pipeline
        response = mock_rag_pipeline.run(request)
        
        # Should return error response
        assert "No user messages" in response.response
    
    def test_pipeline_preserves_conversation_history(self, mock_rag_pipeline):
        """Test that RAG pipeline preserves full conversation history"""
        # Mock retrieval
        mock_retrieval_response = Mock()
        mock_retrieval_response.source_documents = ["Clinical document"]
        mock_rag_pipeline.retrieval_service.retrieve = Mock(return_value=mock_retrieval_response)
        
        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.response = "Continuing conversation..."
        mock_rag_pipeline.chat_model.generate_response = Mock(return_value=mock_chat_response)
        
        # Create request with multi-turn conversation
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Follow-up question")
        ])
        
        # Run pipeline
        response = mock_rag_pipeline.run(request)
        
        # Verify all messages are passed to chat model
        call_args = mock_rag_pipeline.chat_model.generate_response.call_args[0]
        chat_request = call_args[0]
        
        # Should have system message + all original messages
        assert len(chat_request.messages) >= 4  # system + 3 original
        assert any(m.content == "First question" for m in chat_request.messages)
        assert any(m.content == "First answer" for m in chat_request.messages)
        assert any(m.content == "Follow-up question" for m in chat_request.messages)
    
    def test_pipeline_error_handling(self, mock_rag_pipeline):
        """Test error handling in RAG pipeline"""
        # Mock retrieval error
        mock_rag_pipeline.retrieval_service.retrieve = Mock(
            side_effect=Exception("Retrieval error")
        )
        
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="Test question")
        ])
        
        # Run pipeline - should handle error gracefully
        try:
            response = mock_rag_pipeline.run(request)
            # Either returns error response or raises exception
        except Exception as e:
            assert "Retrieval error" in str(e)
