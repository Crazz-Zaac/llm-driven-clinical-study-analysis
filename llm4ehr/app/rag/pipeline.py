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

        print("\n=== LAST USER MESSAGE ===")
        print(last_user_message)

        query_response = self.retrieval_service.retrieve(
            QueryRequest(query=last_user_message)
        )

        docs = query_response.source_documents or []

        retrieved_docs = "\n\n".join(f"""
        [Document: {i+1}]
        Title: {doc.title}
        Article ID: {doc.article_id}
        URL: {doc.url}
        Abstract: {doc.abstract}
        """ for i, doc in enumerate(docs))

        print("\n=== RETRIEVED DOCS ===")
        print(retrieved_docs)

        sys_prompt = f"""
        {SYSTEM_PROMPT}
        You MUST answer using ONLY the context below.
        If the answer is not found, say "Not found in retrieved documents."

        Context:
        {retrieved_docs}
        """

        chat_request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=sys_prompt),
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

        response = self.chat_model.generate_response(chat_request)

        return ChatResponse(
            response=response.response,
            source_documents=[
                doc.model_dump() for doc in query_response.source_documents
            ],
        )
