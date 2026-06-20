from qdrant_client import QdrantClient
from qdrant_client import models
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
                vectors_config={
                    "dense": rest.VectorParams(
                        size=vector_size, distance=rest.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    "sparse": rest.SparseVectorParams(
                        modifier=rest.Modifier.IDF,  # BM25 benefits from IDF reweighting
                    ),
                },
            )

    def dense_search_vectors(
        self, collection_name: str, query_vector: list, top_k: int = 5
    ):
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def hybrid_search_vectors(
        self,
        collection_name: str,
        dense_vector: list,
        sparse_vector: str,
        top_k: int = 5,
    ):
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 10,  # retrieve more for reranking
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                    using="sparse",
                    limit=top_k * 10,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    def delete_collection(self, collection_name: str):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)

    def upsert_vectors(self, collection_name: str, vectors: list):
        self.client.upsert(collection_name=collection_name, points=vectors)

    def search_vectors_filtered(
        self, collection_name: str, query_vector: list, article_id: str, top_k: int = 15
    ):
        """Search within a specific article only."""
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
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

    def hybrid_search_vectors_filtered(
        self,
        collection_name: str,
        dense_vector: list,
        sparse_vector: dict,
        article_id: str,
        top_k: int = 5,
    ):
        """Hybrid (dense+sparse) search within a specific article only."""
        article_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="article_id", match=rest.MatchValue(value=article_id)
                )
            ]
        )
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=article_filter,
                    limit=top_k * 10,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector["indices"], values=sparse_vector["values"]
                    ),
                    using="sparse",
                    filter=article_filter,
                    limit=top_k * 10,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
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
