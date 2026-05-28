from app.rag.pipeline import RAGPipeline
from app.rag.services import RetrievalService
from app.rag.llm.remote_chat_model import ChatModel
from app.rag.prompts.system_prompt import SYSTEM_PROMPT
from app.schemas.chat_schema import ChatMessage, ChatRequest, ChatResponse
from app.schemas.retrieval_schema import QueryRequest


class RAGPipelineRemote(RAGPipeline):  # inherit everything, just swap the model
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.chat_model = ChatModel()
