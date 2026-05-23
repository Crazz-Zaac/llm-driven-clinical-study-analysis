import time
import logging
import requests

from app.core.config import settings
from app.db.activate_model import get_active_embedding_model, get_active_embedding_dimension

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, show_progress: bool = True):
        self.model_name = get_active_embedding_model()  # resolved at instantiation
        self.show_progress = show_progress
        self.ollama_url = settings.OLLAMA_URL
        self._embedding_dimension: int | None = get_active_embedding_dimension()  # eager load

    def embed(self, text: str | list[str], show_timing: bool = True) -> list:
        start_time = time.time()

        is_batch = isinstance(text, list)
        num_texts = len(text) if is_batch else 1

        response = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model_name, "input": text},
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]

        # Now safe to check — dimension is guaranteed to be set
        if len(embeddings[0]) != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dimension}, "
                f"got {len(embeddings[0])}"
            )

        elapsed_time = time.time() - start_time
        if show_timing:
            avg_time = elapsed_time / num_texts if num_texts > 0 else 0
            logger.info(
                f"Embedded {num_texts} text(s) in {elapsed_time:.2f}s "
                f"(avg: {avg_time:.4f}s per text)"
            )

        return embeddings if is_batch else embeddings[0]