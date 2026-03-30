from pydantic import BaseModel, Field
from typing import List

class IngestionRequest(BaseModel):
    documents: List[str] = Field(..., description="A list of documents to be ingested into the system.")

class IngestionResponse(BaseModel):
    success: bool = Field(..., description="Indicates whether the ingestion process was successful.")
    metadata: dict = Field(..., description="Contains IDs, vector representations and payload of the ingested documents, or error details if the ingestion failed.")