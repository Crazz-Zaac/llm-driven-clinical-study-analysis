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

    _stop_requested = False

    def __init__(self):
        self.embedder = TextEmbedder()
        self.chunker = TextChunker()
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        self.embeddings_dir = Path(settings.EMBEDDINGS_DIR)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.vector_db.create_collection(
            self.collection_name,
            vector_size=self.embedder.model.get_sentence_embedding_dimension(),
        )

    def index_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        if self._stop_requested:
            return {"success": False, "indexed_count": 0, "embedding_files": []}
        if not documents:
            return {"success": True, "indexed_count": 0, "embedding_files": []}

        points: list[dict[str, Any]] = []
        embedding_files: list[str] = []

        for doc in documents:
            # Save one embedding file per document (for record keeping)
            combined_text = self._build_combined_text(doc)
            doc_with_text = {**doc, "combined_text": combined_text}
            abstract_chunks = self.chunker.split_text(doc.get("abstract", ""))
            if abstract_chunks:
                abstract_embedding = self.embedder.embed(
                    [abstract_chunks[0].page_content], show_timing=False
                )[0]
                file_path = self._save_embedding(doc_with_text, abstract_embedding)
            embedding_files.append(str(file_path))

            # Chunk and embed each section separately
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
            "embedding_files": embedding_files,
        }

    def index_from_fetched(self, article_ids: list[str]) -> dict[str, Any]:
        """Load fetched articles from disk by article ID and index them."""
        documents = self._load_fetched_documents(article_ids)
        return self.index_documents(documents)

    def _save_embedding(self, doc: dict[str, Any], embedding: list[float]) -> Path:
        file_path = self._embedding_file_path(doc["article_id"])
        payload = {
            "article_id": doc.get("article_id", ""),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "methods": doc.get("methods", ""),
            "results": doc.get("results", ""),
            "discussion": doc.get("discussion", ""),
            "conclusion": doc.get("conclusion", ""),
            "combined_text": doc.get("combined_text", ""),
            "embedding": embedding,
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False))
        return file_path

    def _embedding_file_path(self, article_id: str) -> Path:
        file_id = hashlib.md5(article_id.encode()).hexdigest()
        return self.embeddings_dir / f"{file_id}.json"

    def _load_fetched_documents(self, article_ids: list[str]) -> list[dict[str, Any]]:
        fetched_dir = Path(settings.FETCHED_ARTICLES_DIR)
        logger.info(f"Looking for articles in: {fetched_dir}")
        logger.info(f"Article IDs requested: {article_ids}")
        if not fetched_dir.exists():
            return []

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
        IndexingService._stop_requested = True

    def resume_indexing(self):
        """Reset stop signal to allow indexing."""
        IndexingService._stop_requested = False

    def delete_indexed_documents(self, article_ids: list[str]) -> dict[str, Any]:
        """Delete indexed documents by article ID from disk and Qdrant."""
        deleted_files: list[str] = []
        for article_id in article_ids:
            file_path = self._embedding_file_path(article_id)
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))

        self.vector_db.delete_by_article_ids(self.collection_name, article_ids)
        return {
            "success": True,
            "deleted_count": len(article_ids),
            "deleted_files": deleted_files,
        }

    def delete_collection(self):
        """Delete the entire collection from the vector database."""
        self.vector_db.delete_collection(self.collection_name)


_indexing_service: IndexingService | None = None


def get_indexing_service() -> IndexingService:
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = IndexingService()
    return _indexing_service
