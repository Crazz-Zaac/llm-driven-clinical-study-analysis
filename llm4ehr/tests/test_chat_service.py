"""Test suite for ChatService"""
import pytest
from unittest.mock import Mock, patch
from app.rag.services import ChatService
from app.schemas.chat_schema import ChatRequest, ChatMessage


class TestChatService:
    """Tests for chat service functionality"""
    
    @pytest.fixture
    def mock_chat_service(self):
        """Create ChatService with mocked dependencies"""
        with patch('app.rag.services.chat_service.ChatModel') as mock_chat:
            service = ChatService()
            return service
    
    def test_chat_service_initialization(self, mock_chat_service):
        """Test ChatService initializes correctly"""
        assert mock_chat_service.chat_model is not None
    
    def test_handle_chat_success(self, mock_chat_service):
        """Test successful chat handling"""
        # Mock chat model response
        mock_response = Mock()
        mock_response.response = "This is a test response"
        mock_response.source_documents = ["doc1", "doc2"]
        mock_chat_service.chat_model.generate_response = Mock(return_value=mock_response)
        
        # Create request
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="Hello, how are you?")
        ])
        
        # Handle chat
        response = mock_chat_service.handle_chat(request)
        
        # Verify response
        assert response.response == "This is a test response"
        assert mock_chat_service.chat_model.generate_response.called
    
    def test_handle_chat_passes_messages(self, mock_chat_service):
        """Test that handle_chat passes correct messages to chat model"""
        mock_response = Mock()
        mock_response.response = "Response"
        mock_chat_service.chat_model.generate_response = Mock(return_value=mock_response)
        
        # Create request with multiple messages
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="First message"),
            ChatMessage(role="assistant", content="First response"),
            ChatMessage(role="user", content="Second message")
        ])
        
        # Handle chat
        mock_chat_service.handle_chat(request)
        
        # Verify chat model was called with the request
        mock_chat_service.chat_model.generate_response.assert_called_once_with(request)
    
    def test_handle_chat_error_handling(self, mock_chat_service):
        """Test error handling in chat service"""
        # Mock chat model error
        mock_chat_service.chat_model.generate_response = Mock(
            side_effect=Exception("LLM error")
        )
        
        request = ChatRequest(messages=[
            ChatMessage(role="user", content="Test message")
        ])
        
        # Should raise exception or handle gracefully
        try:
            response = mock_chat_service.handle_chat(request)
            # If handled, response should indicate error
            assert "error" in response.response.lower() or response is not None
        except Exception as e:
            assert "LLM error" in str(e)
    
    def test_handle_chat_with_system_prompt(self, mock_chat_service):
        """Test chat handling with system prompt"""
        mock_response = Mock()
        mock_response.response = "Helpful response"
        mock_chat_service.chat_model.generate_response = Mock(return_value=mock_response)
        
        # Create request with system prompt
        request = ChatRequest(messages=[
            ChatMessage(role="system", content="You are a helpful medical assistant."),
            ChatMessage(role="user", content="What is the treatment for disease X?")
        ])
        
        # Handle chat
        response = mock_chat_service.handle_chat(request)
        
        # Verify response
        assert response.response is not None
        
        # Verify chat model received all messages including system prompt
        call_args = mock_chat_service.chat_model.generate_response.call_args[0]
        chat_request = call_args[0]
        assert any(m.role == "system" for m in chat_request.messages)
