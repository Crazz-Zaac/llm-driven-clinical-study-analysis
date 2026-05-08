import uuid
import logging
import time
import hashlib
from qdrant_client.http.models import PointStruct

from app.rag.chunking.splitter import TextChunker
from app.rag.embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.ingestion_schema import IngestionRequest, IngestionResponse
from app.core.config import settings

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting documents into the vector database."""

    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = TextEmbedder(show_progress=True)
        self.vector_db = QdrantVectorDB()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.batch_size = 100
        self.vector_db.create_collection(
            self.collection_name,
            vector_size=self.embedder.model.get_sentence_embedding_dimension(),
        )

    def _upsert_batch(self, batch_vectors: list) -> None:
        """Upsert a batch of vectors to the database."""
        try:
            self.vector_db.upsert_vectors(
                collection_name=self.collection_name, vectors=batch_vectors
            )
            logger.debug(f"Upserted batch of {len(batch_vectors)} vectors")
        except Exception as e:
            logger.error(f"Error upserting batch: {str(e)}")
            raise

    def ingest_documents(self, request: IngestionRequest) -> IngestionResponse:
        """
        Ingest documents into the vector database.

        Args:
            request: IngestionRequest containing documents to ingest

        Returns:
            IngestionResponse with success status and metadata
        """
        start_time = time.time()
        try:
            all_metadata = []
            batch_vectors = []
            total_chunks = 0

            for doc_index, doc in enumerate(request.documents, 1):
                logger.info(f"Processing document {doc_index}/{len(request.documents)}")
                chunks = self.chunker.split_text(doc)

                chunks = self.chunker.split_text(doc)
                texts = [chunk.page_content for chunk in chunks]
                embeddings = self.embedder.embed(texts)

                for chunk, embedding in zip(chunks, embeddings):
                    total_chunks += 1

                    metadata = {
                        "doc_id": (
                            request.doc_id if hasattr(request, "doc_id") else None
                        ),
                        "chunk_id": str(uuid.uuid4()),
                        "text": chunk.page_content,
                        "source": (
                            chunk.metadata.get("source")
                            if hasattr(chunk, "metadata")
                            else None
                        ),
                        "page": (
                            chunk.metadata.get("page")
                            if hasattr(chunk, "metadata")
                            else None
                        ),
                    }
                    all_metadata.append(metadata)

                    point = PointStruct(
                        id=int(
                            hashlib.sha256(
                                (metadata["doc_id"] + metadata["chunk_id"]).encode()
                            ).hexdigest(),
                            16,
                        )
                        % (10**12),
                        vector=embedding,
                        payload=metadata,
                    )

                    batch_vectors.append(point)

                    # Upsert when batch reaches size limit
                    if len(batch_vectors) >= self.batch_size:
                        self._upsert_batch(batch_vectors)
                        batch_vectors = []

            # Upsert remaining vectors
            if batch_vectors:
                self._upsert_batch(batch_vectors)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Ingestion complete: {total_chunks} chunks in {elapsed_time:.2f}s"
            )

            return IngestionResponse(
                success=True,
                metadata={
                    "ingested_count": len(all_metadata),
                    "total_chunks": total_chunks,
                    "elapsed_time": elapsed_time,
                    "details": all_metadata,
                },
            )
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            return IngestionResponse(success=False, metadata={"error": str(e)})
