from app.rag.llm.chat_model import ChatModel
from app.schemas.chat_schema import ChatMessage, ChatRequest
from app.rag.prompts.system_prompt import SYSTEM_PROMPT


def llm_basic_response_test():
    chat_model = ChatModel()

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content="What is 2 + 2?"),
    ]

    request = ChatRequest(messages=messages)

    response = chat_model.generate_response(request)

    print("LLM Response:", response)

    assert response is not None
    assert len(response) > 0

if __name__ == "__main__":
    llm_basic_response_test()