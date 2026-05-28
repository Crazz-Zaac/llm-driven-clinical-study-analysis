from app.rag.services import RetrievalService
from app.rag.llm.ollama_chat_model import OllamaChatModel
from app.rag.prompts.system_prompt import SYSTEM_PROMPT
from app.schemas.chat_schema import ChatMessage, ChatRequest, ChatResponse
from app.schemas.query_schema import QueryRequest


class RAGPipeline:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        # this is the local model used for generating responses in the RAG pipeline, it can be swapped out if needed
        self.chat_model = OllamaChatModel()

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
        if not docs:
            return ChatResponse(
                response="Not found in retrieved documents. Please ensure the article is indexed.",
                source_documents=[],
            )

        retrieved_docs = "\n\n".join(f"""
        [Document {i+1}]
        Title: {doc.title}
        Article ID: {doc.article_id}
        Section: {doc.section}
        Content: {doc.combined_text}
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
                doc.model_dump(exclude=["abstract"])
                for doc in (query_response.source_documents or [])
            ],
        )
