import hashlib
from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
import json
from pathlib import Path

# Load JSON
data_dir = Path(__file__).parent.parent / "app" / "data"

file_lists = list(data_dir.glob("*.json"))

for file in file_lists:
    with open(file, "r") as f:
        article = json.load(f)

    # Initialize embedder and DB
    embedder = TextEmbedder()
    db = QdrantVectorDB()

    # Create collection if needed
    db.create_collection("articles", vector_size=384)  # all-MiniLM-L6-v2 = 384 dims

    # Store in Qdrant
    from qdrant_client.http.models import PointStruct

    combined_text = f"""
        Title: {article.get("title", "")}

        Abstract:
        {article.get("abstract", "")}

        Methods:
        {article.get("methods", "")}

        Results:
        {article.get("results", "")}    
    """

    vector = embedder.embed(
        combined_text
    )  # Get embedding vector for the abstract and methodology

    point = PointStruct(
        id=int(hashlib.md5(article["article_id"].encode()).hexdigest(), 16) % (10**12),
        vector=vector,
        payload={
            "article_id": article["article_id"],
            "url": article["url"],
            "title": article.get("title", ""),
            "abstract": article.get("abstract", ""),
            "combined_text": combined_text,
        },
    )
    db.upsert_vectors("articles", [point])
    print(f"Processed and stored article: {article['article_id']}")
