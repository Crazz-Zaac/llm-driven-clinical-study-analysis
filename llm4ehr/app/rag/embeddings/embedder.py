from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)

    def embed(self, text: str):
        return self.model.encode(text).tolist()