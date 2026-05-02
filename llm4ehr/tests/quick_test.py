from huggingface_hub import InferenceClient
from app.core.config import settings

class ChatModel:
    def __init__(self):
        self.client = InferenceClient(
            model=settings.HF_MODEL,
            token=settings.HF_API_KEY,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    chat_model = ChatModel()
    prompt = "What is the capital of France?"
    response = chat_model.generate(prompt)
    print("LLM Response:", response)