from app.schemas.chat_schema import ChatRequest, ChatResponse, ChatMessage
from app.core.config import settings
from typing import List
from huggingface_hub import InferenceClient


class ChatModel:
    def __init__(self):
        """Initialize the chat model with HuggingFace LLM."""
        self.client = InferenceClient(
            model=settings.HF_MODEL,
            token=settings.HF_API_KEY,
        )

        if not settings.HF_API_KEY:
            raise ValueError("HF_API_KEY environment variable is not set")

    def generate_response(self, request: ChatRequest) -> ChatResponse:
        """Generate a response using the LLM with chat history and system prompt."""
        try:
            if not request.messages:
                return ChatResponse(
                    response="No messages found in the conversation history.",
                    source_documents=None,
                )

            # Call the LLM to generate a response
            response_text = (
                self.client.chat_completion(
                    messages=[
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ]
                )
                .choices[0]
                .message.content
            )

            return ChatResponse(
                response=response_text,
                source_documents=None,
            )
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            return ChatResponse(
                response=f"Error generating response: {repr(e)}", source_documents=None
            )
