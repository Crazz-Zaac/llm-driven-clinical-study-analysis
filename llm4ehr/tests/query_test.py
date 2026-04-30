from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
import json
from pathlib import Path

from app.db.qdrant_client import QdrantVectorDB


class TestQuery:
    def setup_method(self):
        self.embedder = TextEmbedder()
        self.db = QdrantVectorDB()

    def test_query_qdrant(self):
        # query = "Impact of anti-T-lymphocyte"
        query = "extensive cGVHD risk"

        query_vector = self.embedder.embed(query)

        results = self.db.search_vectors(
            collection_name="articles", query_vector=query_vector, top_k=5
        )

        points = results.points
        for i, p in enumerate(points):
            print(f"Result {i+1}")
            print("Score:", p.score)
            print("Article ID:", p.payload.get("article_id"))
            print("Title:", p.payload.get("title"))
            print("URL:", p.payload.get("url"))
            print("-----")

        assert len(points) > 0, "No results found"
