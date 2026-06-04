import requests
import json
from app.core.config import settings

OLLAMA_URL = settings.OLLAMA_URL
MAX_MODEL_SIZE = settings.MAX_MODEL_SIZE


class OllamaService:

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        response = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model_name})
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_model_size_bytes(model_name: str) -> int | None:
        """
        Returns disk size in bytes from /api/tags if the model is already
        pulled, otherwise falls back to parsing parameter_size from /api/show.
        Returns None if size cannot be determined.
        """
        # Check local models first — /api/tags has exact byte size
        tags = requests.get(f"{OLLAMA_URL}/api/tags")
        tags.raise_for_status()
        for model in tags.json().get("models", []):
            if model["name"] == model_name or model["model"] == model_name:
                return model["size"]  # exact bytes, e.g. 4683075271

        # Model not yet pulled — estimate from parameter_size string in /api/show
        try:
            info = OllamaService.get_model_info(model_name)
            param_size = info.get("details", {}).get("parameter_size", "")
            return OllamaService._parse_param_size(param_size)
        except requests.HTTPError:
            return None  # model not found in registry

    @staticmethod
    def _parse_param_size(param_size: str) -> int | None:
        """
        Converts strings like '7.6B', '70B', '134.52M' to estimated bytes.
        Assumes Q4 quantization (~0.5 bytes/param) as a conservative estimate.
        """
        if not param_size:
            return None
        param_size = param_size.upper().strip()
        multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
        for suffix, mult in multipliers.items():
            if param_size.endswith(suffix):
                try:
                    count = float(param_size[:-1])
                    return int(count * mult * 0.5)  # ~0.5 bytes/param at Q4
                except ValueError:
                    return None
        return None

    @staticmethod
    def get_embedding_dimension(model_name: str) -> int:
        response = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        response.raise_for_status()
        model_info = response.json().get("model_info", {})
        for k, v in model_info.items():
            if "embedding" in k.lower() and "length" in k.lower():
                return v
        raise ValueError(
            f"Embedding dimension not found in model metadata for '{model_name}'"
        )

    @staticmethod
    def pull_model(model_name: str):
        size = OllamaService.get_model_size_bytes(model_name)

        if size is not None and size > MAX_MODEL_SIZE:
            raise ValueError(
                f"Model '{model_name}' is ~{size / 1e9:.1f} GB, "
                f"exceeds limit of {MAX_MODEL_SIZE / 1e9:.1f} GB."
            )
        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name},
            timeout=600,
        )
        response.raise_for_status()
        lines = [line for line in response.text.strip().splitlines() if line]
        last = json.loads(lines[-1])

        if last.get("status") != "success":
            raise ValueError(f"Pull did not complete successfully: {last}")

        return last

    @staticmethod
    def pull_model_stream(model_name: str):
        size = OllamaService.get_model_size_bytes(model_name)

        if size is not None and size > MAX_MODEL_SIZE:
            raise ValueError(
                f"Model '{model_name}' is ~{size / 1e9:.1f} GB, "
                f"exceeds limit of {MAX_MODEL_SIZE / 1e9:.1f} GB."
            )

        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name},
            timeout=600,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

    @staticmethod
    def list_models():
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        return response.json()

    @staticmethod
    def delete_model(model_name: str):
        response = requests.delete(
            f"{OLLAMA_URL}/api/delete",
            json={"model": model_name},  # changed "name" → "model"
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
