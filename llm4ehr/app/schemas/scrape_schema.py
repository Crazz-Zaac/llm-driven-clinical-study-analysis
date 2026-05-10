from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict
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
    scraped_at: datetime
    sections: SectionSchema
    metadata: Optional[MetadataSchema] = None


class ScrapTextRequest(BaseModel):
    url: HttpUrl


class ScrapTextBatchRequest(BaseModel):
    urls: List[HttpUrl]


class ScrapTextResponse(BaseModel):
    article: ArticleSchema
    message: Optional[str] = "Article scraped successfully"


class ScrapTextBatchResponse(BaseModel):
    articles: List[ArticleSchema]
    message: Optional[str] = "Articles scraped successfully"


class ScrapTextListItem(BaseModel):
    article_id: str
    title: Optional[str] = None
    url: HttpUrl


class ScrapTextListResponse(BaseModel):
    articles: List[ScrapTextListItem]


class ScrapTextErrorResponse(BaseModel):
    error: str
