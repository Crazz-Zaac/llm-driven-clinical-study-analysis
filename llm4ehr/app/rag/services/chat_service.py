from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.rag.llm.remote_chat_model import ChatModel
from app.rag.prompts.system_prompt import SYSTEM_PROMPT


class ChatService:
    def __init__(self):
        self.chat_model = ChatModel()

    def handle_chat(self, request: ChatRequest) -> ChatResponse:
        # Generate response using the chat model
        response = self.chat_model.generate_response(request)
        return response
