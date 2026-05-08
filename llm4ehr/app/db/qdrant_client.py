from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.core.config import settings


class QdrantVectorDB:
    """Class to manage interactions with Qdrant vector database"""

    def __init__(self):
        """Initialize Qdrant client using settings from environment variables"""
        self.client = QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )

    def create_collection(self, collection_name: str, vector_size: int):
        """Create a new collection in Qdrant if it doesn't exist"""
        if not self.client.collection_exists(collection_name):
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size, distance=rest.Distance.COSINE
                ),
            )

    def upsert_vectors(self, collection_name: str, vectors: list):
        """Upsert a list of vectors into the specified collection"""
        self.client.upsert(collection_name=collection_name, points=vectors)

    def search_vectors(self, collection_name: str, query_vector: list, top_k: int = 5):
        """Search for similar vectors in the specified collection"""
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        # returning the points with payload for further processing in the retrieval service
        return response.points

    def delete_collection(self, collection_name: str):
        """Delete an entire collection from Qdrant"""
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)

    def delete_by_article_ids(
        self, collection_name: str, article_ids: list[str]
    ) -> None:
        """Delete points matching article IDs from a collection"""
        if not article_ids:
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="article_id",
                        match=rest.MatchAny(any=article_ids),
                    )
                ]
            ),
        )
