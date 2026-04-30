from app.rag.services import RetrievalService
from app.rag.llm.chat_model import ChatModel
from app.rag.prompts.system_prompt import SYSTEM_PROMPT
from app.schemas.chat_schema import ChatMessage, ChatRequest, ChatResponse
from app.schemas.query_schema import QueryRequest


class RAGPipeline:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.chat_model = ChatModel()

    def run(self, request: ChatRequest) -> ChatResponse:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return ChatResponse(
                response="No user messages found in the conversation history.",
                source_documents=None,
            )

        last_user_message = user_messages[-1].content

        query_response = self.retrieval_service.retrieve(
            QueryRequest(query=last_user_message)
        )

        retrieved_docs = "\n\n".join(f"""
    Title: {doc.get('title', '')}
    Article ID: {doc.get('article_id', '')}
    URL: {doc.get('url', '')}
    Abstract: {doc.get('abstract', '')}
    """ for doc in (query_response.source_documents or []))

        chat_request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"""
                    Context:
                    {retrieved_docs}

                    Question:
                    {last_user_message}
                    """,
                ),
            ]
        )

        return self.chat_model.generate_response(chat_request)
