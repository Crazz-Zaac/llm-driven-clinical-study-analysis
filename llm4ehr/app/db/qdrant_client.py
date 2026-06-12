from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from app.core.config import settings


class QdrantVectorDB:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )

    def create_collection(self, collection_name: str, vector_size: int):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size, distance=rest.Distance.COSINE
                ),
            )

    def delete_collection(self, collection_name: str):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)

    def upsert_vectors(self, collection_name: str, vectors: list):
        self.client.upsert(collection_name=collection_name, points=vectors)

    def search_vectors(self, collection_name: str, query_vector: list, top_k: int = 5):
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def search_vectors_filtered(
        self, collection_name: str, query_vector: list, article_id: str, top_k: int = 15
    ):
        """Search within a specific article only."""
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="article_id", match=rest.MatchValue(value=article_id)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def find_article_id_by_title(
        self, collection_name: str, title_fragment: str
    ) -> str | None:
        """Find article_id by partial title match."""
        results, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="title", match=rest.MatchText(text=title_fragment)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return results[0].payload.get("article_id") if results else None

    def delete_by_article_ids(self, collection_name: str, article_ids: list[str]):
        if not article_ids:
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="article_id", match=rest.MatchAny(any=article_ids)
                    )
                ]
            ),
        )
