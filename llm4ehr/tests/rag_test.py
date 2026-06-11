from app.rag.pipeline import RAGPipeline
from app.schemas.chat_schema import ChatRequest, ChatMessage


def run_rag():
    pipeline = RAGPipeline()

    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Which algorithm was used for feature selection in the paper 'Association between triglyceride-glucose index and in-hospital mortality in critically ill patients with sepsis: analysis of the MIMIC-IV database'?"),
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