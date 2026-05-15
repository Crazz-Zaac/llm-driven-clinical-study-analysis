import hashlib
import json
from pathlib import Path
from typing import Any

from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.core.config import settings


class IndexingService:
    """Service for indexing documents into the vector database."""

    _stop_requested = False

    def __init__(self, embeddings_dir: Path | None = None):
        self.embedder = TextEmbedder()
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        base_dir = Path(__file__).resolve().parents[3]
        configured_dir = Path(embeddings_dir or settings.EMBEDDINGS_DIR)
        self.embeddings_dir = (
            configured_dir
            if configured_dir.is_absolute()
            else base_dir / configured_dir
        )
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.vector_db.create_collection(
            self.collection_name,
            vector_size=self.embedder.model.get_sentence_embedding_dimension(),
        )

    def index_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Index a list of documents into the vector database and save embeddings to disk."""
        if self._stop_requested:
            return {"success": False, "indexed_count": 0, "embedding_files": []}
        if not documents:
            return {"success": True, "indexed_count": 0, "embedding_files": []}

        combined_texts: list[str] = []
        prepared_docs: list[dict[str, Any]] = []
        for doc in documents:
            combined_text = self._build_combined_text(doc)
            prepared_doc = {**doc, "combined_text": combined_text}
            prepared_docs.append(prepared_doc)
            combined_texts.append(combined_text)

        embeddings = self.embedder.embed(combined_texts, show_timing=False)

        points: list[dict[str, Any]] = []
        embedding_files: list[str] = []
        for doc, embedding in zip(prepared_docs, embeddings):
            if self._stop_requested:
                break
            file_path = self._save_embedding(doc, embedding)
            embedding_files.append(str(file_path))

            point = {
                "id": self._make_point_id(doc["article_id"]),
                "vector": embedding,
                "payload": {
                    "article_id": doc.get("article_id", ""),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "combined_text": doc.get("combined_text", ""),
                },
            }
            points.append(point)

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

    def _build_combined_text(self, doc: dict[str, Any]) -> str:
        return (
            "Title: {title}\n\n"
            "Abstract:\n{abstract}\n\n"
            "Methods:\n{methods}\n\n"
            "Results:\n{results}\n"
            "Conclusion:\n{conclusion}"
        ).format(
            title=doc.get("title", ""),
            abstract=doc.get("abstract", ""),
            methods=doc.get("methods", ""),
            results=doc.get("results", ""),
            conclusion=doc.get("conclusion", ""),
        )

    def _save_embedding(self, doc: dict[str, Any], embedding: list[float]) -> Path:
        file_path = self._embedding_file_path(doc["article_id"])
        payload = {
            "article_id": doc.get("article_id", ""),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "methods": doc.get("methods", ""),
            "results": doc.get("results", ""),
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
        if not fetched_dir.is_absolute():
            fetched_dir = Path(__file__).resolve().parents[3] / fetched_dir
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
