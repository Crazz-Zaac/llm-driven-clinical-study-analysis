from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
import json
from pathlib import Path

# Load JSON
data_dir = Path(__file__).parent.parent / "app" / "data"

with open(data_dir / "s41409-025-02761-5_20260418_111849_67214a49.json") as f:
    article = json.load(f)

# Initialize embedder and DB
embedder = TextEmbedder()
db = QdrantVectorDB()

# Embed the abstract
embedding = embedder.embed(article['abstract'])

# Create collection if needed
db.create_collection("articles", vector_size=384)  # all-MiniLM-L6-v2 = 384 dims

# Store in Qdrant
from qdrant_client.http.models import PointStruct
point = PointStruct(
    id=hash(article['article_id']),
    vector=embedding[0],
    payload={
        "article_id": article['article_id'],
        "url": article['url'],
        "title": article.get('title', '')
    }
)
db.upsert_vectors("articles", [point])