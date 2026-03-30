from app.rag.retrieval.query import QueryService
from app.rag.llm.chat_model import ChatModel
from app.rag.prompts.system_prompt import SYSTEM_PROMPT
from app.schemas.chat_schema import ChatMessage, ChatRequest, ChatResponse
from llm4ehr.app.schemas.query_schema import QueryRequest

class RAGPipeline:
    def __init__(self):
        self.query_service = QueryService()
        self.chat_model = ChatModel()

    def run(self, request: ChatRequest) -> ChatResponse:
        # Step 1: Process the last user message as a query
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return ChatResponse(response="No user messages found in the conversation history.", source_documents=None)
        
        last_user_message = user_messages[-1].content
        query_response = self.query_service.process_query(QueryRequest(query=last_user_message))

        # Step 2: Generate a response using the chat model with retrieved documents
        retrieved_docs = "\n".join(query_response.source_documents) if query_response.source_documents else "No relevant documents found."
        system_prompt_filled = SYSTEM_PROMPT.format(retrieved_docs=retrieved_docs, question=last_user_message)
        
        # Create a new ChatRequest for the chat model with the system prompt and conversation history
        chat_request = ChatRequest(messages=[
            ChatMessage(role="system", content=system_prompt_filled),
            *request.messages
        ])
        
        return self.chat_model.generate_response(chat_request)