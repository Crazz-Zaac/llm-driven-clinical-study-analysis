import requests

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.schemas.chat_schema import ChatRequest, ChatResponse, HostedProvider


OPENAI_COMPATIBLE_PROVIDERS = {
    HostedProvider.OPENAI: {
        "label": "OpenAI",
        "base_url": lambda: settings.OPENAI_BASE_URL,
    },
    HostedProvider.MISTRAL: {
        "label": "Mistral AI",
        "base_url": lambda: settings.MISTRAL_BASE_URL,
    },
}


class ChatModel:
    def __init__(self):
        """Initialize a request-configurable hosted chat model client."""

    def _request_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

    def _resolve_provider(self, request: ChatRequest) -> HostedProvider:
        return request.hosted_provider or HostedProvider.HUGGINGFACE

    def _resolve_api_key(self, request: ChatRequest) -> str:
        return request.hosted_api_key.strip() if request.hosted_api_key else ""

    def _resolve_model_name(self, request: ChatRequest) -> str:
        return request.hosted_model_name.strip() if request.hosted_model_name else ""

    def _generate_huggingface_response(
        self,
        request: ChatRequest,
        model_name: str,
        api_key: str,
    ) -> str:
        if not api_key:
            raise ValueError("hosted_api_key is required for Hugging Face")
        if not model_name:
            raise ValueError("hosted_model_name is required for Hugging Face")

        client = InferenceClient(model=model_name, token=api_key)
        return (
            client.chat_completion(
                messages=self._request_messages(request),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            .choices[0]
            .message.content
        )

    def _generate_openai_compatible_response(
        self,
        request: ChatRequest,
        provider: HostedProvider,
        model_name: str,
        api_key: str,
    ) -> str:
        provider_config = OPENAI_COMPATIBLE_PROVIDERS[provider]
        provider_label = provider_config["label"]
        base_url = provider_config["base_url"]().rstrip("/")

        if not api_key:
            raise ValueError(f"hosted_api_key is required for {provider_label}")
        if not model_name:
            raise ValueError(f"hosted_model_name is required for {provider_label}")

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": self._request_messages(request),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            timeout=120,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"{provider_label} API error {response.status_code}: "
                f"{response.text[:500]}"
            )

        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def generate_response(self, request: ChatRequest) -> ChatResponse:
        """Generate a response using the selected hosted chat provider."""
        try:
            if not request.messages:
                return ChatResponse(
                    response="No messages found in the conversation history.",
                    source_documents=None,
                )

            provider = self._resolve_provider(request)
            model_name = self._resolve_model_name(request)
            api_key = self._resolve_api_key(request)

            if provider == HostedProvider.HUGGINGFACE:
                response_text = self._generate_huggingface_response(
                    request,
                    model_name,
                    api_key,
                )
            elif provider in OPENAI_COMPATIBLE_PROVIDERS:
                response_text = self._generate_openai_compatible_response(
                    request,
                    provider,
                    model_name,
                    api_key,
                )
            else:
                raise ValueError(f"Hosted provider '{provider.value}' is not supported")

            return ChatResponse(
                response=response_text,
                source_documents=None,
            )
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            return ChatResponse(
                response=f"Error generating response: {repr(e)}", source_documents=None
            )
