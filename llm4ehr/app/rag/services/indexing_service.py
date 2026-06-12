import hashlib
import json
from pathlib import Path
from typing import Any
from loguru import logger

from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.rag.chunking.splitter import TextChunker
from app.core.config import settings


class IndexingService:
    """Service for indexing documents into the vector database."""

    def __init__(self):
        self._stop_requested = False
        self.embedder = TextEmbedder()
        self.chunker = TextChunker()
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def index_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        self.vector_db.create_collection(
            self.collection_name, vector_size=self.embedder._embedding_dimension
        )
        if self._stop_requested:
            return {"success": False, "indexed_count": 0, "embedding_files": []}
        if not documents:
            return {"success": True, "indexed_count": 0, "embedding_files": []}

        points: list[dict[str, Any]] = []

        for doc in documents:
            for section_name in [
                "abstract",
                "methods",
                "results",
                "discussion",
                "conclusion",
            ]:
                section_text = doc.get(section_name, "")
                if not section_text:
                    continue

                chunks = self.chunker.split_text(section_text)
                chunk_texts = [chunk.page_content for chunk in chunks]
                if not chunk_texts:
                    continue

                embeddings = self.embedder.embed(chunk_texts, show_timing=False)

                for i, (chunk_text, embedding) in enumerate(
                    zip(chunk_texts, embeddings)
                ):
                    if self._stop_requested:
                        break
                    points.append(
                        {
                            "id": self._make_point_id(
                                f"{doc['article_id']}_{section_name}_{i}"
                            ),
                            "vector": embedding,
                            "payload": {
                                "article_id": doc.get("article_id", ""),
                                "url": doc.get("url", ""),
                                "title": doc.get("title", ""),
                                "abstract": doc.get("abstract", ""),
                                "section": section_name,
                                "combined_text": chunk_text,
                                "chunk_index": i,
                            },
                        }
                    )

        if points:
            self.vector_db.upsert_vectors(self.collection_name, points)

        return {
            "success": True,
            "indexed_count": len(points),
        }

    def index_from_fetched(self, article_ids: list[str]) -> dict[str, Any]:
        """Load fetched articles from disk by article ID and index them."""
        documents = self._load_fetched_documents(article_ids)
        return self.index_documents(documents)

    def _load_fetched_documents(self, article_ids: list[str]) -> list[dict[str, Any]]:
        fetched_dir = Path(settings.FETCHED_ARTICLES_DIR)
        logger.info(f"Looking for articles in: {fetched_dir}")
        logger.info(f"Article IDs requested: {article_ids}")
        if not fetched_dir.exists():
            return []

        # if no specific article IDs are provided, load all fetched articles
        if not article_ids:
            all_files = sorted(
                fetched_dir.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            logger.info(f"Found {len(all_files)} fetched article files")
            documents = []
            for file_path in all_files:
                with open(file_path, "r", encoding="utf-8") as handle:
                    documents.append(json.load(handle))
            return documents

        # Load only the specified article IDs, using the most recent file if multiple exist for the same ID
        documents: list[dict[str, Any]] = []
        for article_id in article_ids:
            matches = sorted(
                fetched_dir.glob(f"{article_id}_*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                continue
            with open(matches[0], "r", encoding="utf-8") as handle:
                documents.append(json.load(handle))

        return documents

    def _make_point_id(self, article_id: str) -> int:
        return int(hashlib.md5(article_id.encode()).hexdigest(), 16) % (10**12)

    def stop_indexing(self):
        """Signal the indexing loop to stop."""
        self._stop_requested = True

    def resume_indexing(self):
        """Reset stop signal to allow indexing."""
        self._stop_requested = False

    # deletes the collection and all its data from the vector database, and then recreates it to ensure a clean state
    def delete_collection(self):
        """Delete and recreate the collection in the vector database."""
        self.vector_db.delete_collection(self.collection_name)
        if self.vector_db.client.collection_exists(self.collection_name):
            logger.warning(
                f"Collection '{self.collection_name}' still exists after deletion attempt"
            )
        else:
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
        self.vector_db.create_collection(
            self.collection_name, vector_size=self.embedder._embedding_dimension
        )
        logger.info(f"Collection '{self.collection_name}' recreated successfully")


_indexing_service: IndexingService | None = None


def get_indexing_service() -> IndexingService:
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = IndexingService()
    return _indexing_service
