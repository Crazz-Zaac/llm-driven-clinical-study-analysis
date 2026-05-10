from pydantic import BaseModel, Field
from typing import List


class IndexDocument(BaseModel):
    article_id: str = Field(..., description="Unique article identifier")
    url: str = Field(..., description="Source URL")
    title: str = Field("", description="Document title")
    abstract: str = Field("", description="Document abstract")
    methods: str = Field("", description="Methods section")
    results: str = Field("", description="Results section")
    conclusion: str = Field("", description="Conclusion section")


class IndexRequest(BaseModel):
    documents: List[IndexDocument] = Field(..., description="Documents to index")


class IndexFromScrapedRequest(BaseModel):
    article_ids: List[str] = Field(
        default_factory=list,
        description="Article IDs to load from data/scrapped_articles",
    )


class IndexResponse(BaseModel):
    success: bool = Field(..., description="Whether indexing succeeded")
    indexed_count: int = Field(..., description="Number of documents indexed")
    embedding_files: List[str] = Field(
        ..., description="Paths to saved embedding files"
    )


class DeleteIndexRequest(BaseModel):
    article_ids: List[str] = Field(
        default_factory=list, description="Article IDs to delete"
    )
    delete_all: bool = Field(
        False, description="If true, deletes entire collection and all files"
    )


class DeleteIndexResponse(BaseModel):
    success: bool = Field(..., description="Whether delete succeeded")
    deleted_count: int = Field(..., description="Number of documents deleted")
    deleted_files: List[str] = Field(
        ..., description="Paths to deleted embedding files"
    )
