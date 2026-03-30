from app.schemas.chat_schema import ChatRequest, ChatResponse, ChatMessage
from app.core.config import settings
from typing import List

try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    HuggingFaceEndpoint = None

class ChatModel:
    def __init__(self):
        """Initialize the chat model with HuggingFace LLM."""
        if HuggingFaceEndpoint is None:
            raise ImportError("langchain_huggingface is required. Install it with: pip install langchain-huggingface")
        
        if not settings.HF_API_KEY:
            raise ValueError("HF_API_KEY environment variable is not set")
        
        self.llm = HuggingFaceEndpoint(
            repo_id=settings.HF_MODEL,
            huggingfacehub_api_token=settings.HF_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_new_tokens=settings.LLM_MAX_TOKENS,
        )

    def _format_messages_for_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string for the LLM."""
        formatted_parts = []
        
        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")
        
        # Add a prompt marker for the assistant to respond
        formatted_parts.append("Assistant:")
        return "\n".join(formatted_parts)

    def generate_response(self, request: ChatRequest) -> ChatResponse:
        """Generate a response using the LLM with chat history and system prompt."""
        try:
            if not request.messages:
                return ChatResponse(response="No messages found in the conversation history.", source_documents=None)
            
            # Format all messages (including chat history and system prompt) into a prompt
            prompt = self._format_messages_for_prompt(request.messages)
            
            # Call the LLM to generate a response
            response_text = self.llm.invoke(prompt)
            
            # Extract source documents if present in the last message
            source_documents = None
            
            return ChatResponse(response=response_text, source_documents=source_documents)
        
        except Exception as e:
            return ChatResponse(response=f"Error generating response: {str(e)}", source_documents=None)