from app.rag.pipeline import RAGPipeline
from app.schemas.chat_schema import ChatRequest, ChatMessage


def run_rag():
    pipeline = RAGPipeline()

    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="What is Anti-T-lymphocyte globulin (ATLG)?"),
        ]
    )

    response = pipeline.run(request)

    print("\n=== RAG RESPONSE ===\n")
    print(response.response)

    print("\n=== SOURCES ===\n")
    for doc in response.source_documents or []:
        print(doc)


if __name__ == "__main__":
    run_rag()