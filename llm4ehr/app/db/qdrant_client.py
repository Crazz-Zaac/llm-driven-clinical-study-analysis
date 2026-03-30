from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.core import config

class QdrantVectorDB:
    """Class to manage interactions with Qdrant vector database"""

    def __init__(self):
        """Initialize Qdrant client using settings from environment variables"""
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )

    def create_collection(self, collection_name: str, vector_size: int):
        """Create a new collection in Qdrant if it doesn't exist"""
        if not self.client.has_collection(collection_name):
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
            )
    
    def upsert_vectors(self, collection_name: str, vectors: list):
        """Upsert a list of vectors into the specified collection"""
        self.client.upsert(
            collection_name=collection_name,
            points=vectors
        )

    def search_vectors(self, collection_name: str, query_vector: list, top_k: int = 5):
        """Search for similar vectors in the specified collection"""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

