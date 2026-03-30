import uuid

from chunking.splitter import TextChunker
from embeddings.embedder import TextEmbedder
from app.db.qdrant_client import QdrantVectorDB
from app.schemas.ingestion_schema import IngestionRequest, IngestionResponse

class IngestionService:
    """
    Docstring for IngestionService
    """

    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = TextEmbedder()
        self.vector_db = QdrantVectorDB()
        self.collection_name = "research_documents"
        self.vector_db.create_collection(self.collection_name, vector_size=self.embedder.embedding_dim)

    def ingest_documents(self, request: IngestionRequest) -> IngestionResponse:
        try:
            all_metadata = []
            batch_vectors = []
            batch_size = 100  # Number of vectors to insert per batch
            
            for doc in request.documents:
                chunks = self.chunker.split_text(doc)
                for chunk in chunks:
                    embedding = self.embedder.get_embedding(chunk.page_content)
                    metadata = {
                        "doc_id": request.doc_id if hasattr(request, "doc_id") else None,
                        "chunk_id": str(uuid.uuid4()),
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source"),
                        "page": chunk.metadata.get("page"),
                    }
                    all_metadata.append(metadata)
                    
                    # Add vector to batch
                    batch_vectors.append({
                        "chunk_id": metadata["chunk_id"],
                        "vector": embedding,
                        "payload": metadata
                    })
                    
                    # Upsert when batch reaches size limit
                    if len(batch_vectors) >= batch_size:
                        self.vector_db.upsert_vectors(
                            collection_name=self.collection_name,
                            vectors=batch_vectors
                        )
                        batch_vectors = []
            
            # Upsert remaining vectors
            if batch_vectors:
                self.vector_db.upsert_vectors(
                    collection_name=self.collection_name,
                    vectors=batch_vectors
                )
            
            return IngestionResponse(success=True, metadata={"ingested_count": len(all_metadata), "details": all_metadata})
        except Exception as e:
            return IngestionResponse(success=False, metadata={"error": str(e)})
