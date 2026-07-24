from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from datetime import datetime


class SectionSchema(BaseModel):
    abstract: Optional[str] = None
    methods: Optional[str] = None
    results: Optional[str] = None
    conclusion: Optional[str] = None


class MetadataSchema(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    journal: Optional[str] = None
    published_date: Optional[str] = None


class ArticleSchema(BaseModel):
    article_id: str
    url: HttpUrl
    source: str = "nature"
    fetched_at: datetime
    sections: SectionSchema
    metadata: Optional[MetadataSchema] = None


class FetchTextRequest(BaseModel):
    url: HttpUrl


class FetchTextBatchRequest(BaseModel):
    urls: List[HttpUrl]


class FetchTextResponse(BaseModel):
    article: ArticleSchema
    message: Optional[str] = "Article fetched successfully"


class FetchTextBatchFailure(BaseModel):
    url: HttpUrl
    error: str


class FetchTextBatchResponse(BaseModel):
    articles: List[ArticleSchema]
    failed_urls: List[FetchTextBatchFailure] = Field(default_factory=list)
    message: Optional[str] = "Articles fetched successfully"


class FetchTextListItem(BaseModel):
    article_id: str
    title: Optional[str] = None
    url: HttpUrl


class FetchTextListResponse(BaseModel):
    articles: List[FetchTextListItem]


class DeleteFetchedArticlesRequest(BaseModel):
    article_ids: List[str] = Field(default_factory=list)
    article_urls: List[str] = Field(default_factory=list)
    delete_all: bool = False


class DeleteFetchedArticlesResponse(BaseModel):
    success: bool
    deleted_count: int
    deleted_files: List[str]


class FetchTextErrorResponse(BaseModel):
    error: str
