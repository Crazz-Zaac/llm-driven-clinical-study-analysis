import requests


from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.core.config import settings
from app.db.activate_model import get_active_chat_model
from app.rag.services.ollama_service import OllamaService


class OllamaChatModel:
    def generate_response(self, request: ChatRequest) -> ChatResponse:
        """Generate a response using the local Ollama LLM."""
        try:
            if not request.messages:
                return ChatResponse(
                    response="No messages found in the conversation history.",
                    source_documents=None,
                )

            model_name = get_active_chat_model()  # raises if none activated

            payload = {
                "model": model_name,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "stream": False,
            }

            response = requests.post(
                f"{settings.OLLAMA_URL}/api/chat",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")

            return ChatResponse(response=content, source_documents=None)

        except Exception as e:
            return ChatResponse(
                response=f"Error generating response: {repr(e)}",
                source_documents=None,
            )
