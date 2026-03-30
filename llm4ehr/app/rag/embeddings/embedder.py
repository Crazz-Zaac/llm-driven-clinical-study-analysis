from sentence_transformers import SentenceTransformer
import time
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", show_progress: bool = True):
        self.model = SentenceTransformer(embedding_model_name)
        self.show_progress = show_progress

    def embed(self, text: str, show_timing: bool = True):
        """
        Embed text(s) and optionally show timing information.
        
        Args:
            text: String or list of strings to embed
            show_timing: Whether to log timing information
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        start_time = time.time()
        
        # Determine if input is batch or single
        is_batch = isinstance(text, list)
        num_texts = len(text) if is_batch else 1
        
        # Embed with progress bar (only shown outside pytest)
        embeddings = self.model.encode(
            sentences=text,
            show_progress_bar=self.show_progress and not self._in_test_environment()
        )
        
        elapsed_time = time.time() - start_time
        
        # Log timing information
        if show_timing:
            avg_time = elapsed_time / num_texts if num_texts > 0 else 0
            logger.info(
                f"Embedded {num_texts} text(s) in {elapsed_time:.2f}s "
                f"(avg: {avg_time:.4f}s per text)"
            )
        
        # Return as list (single embedding or list of embeddings)
        if is_batch:
            return embeddings.tolist()
        else:
            return embeddings.tolist()
    
    def _in_test_environment(self) -> bool:
        """Check if running in pytest environment"""
        import sys
        return 'pytest' in sys.modules