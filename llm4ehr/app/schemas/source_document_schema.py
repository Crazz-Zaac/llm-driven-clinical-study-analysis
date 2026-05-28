from pydantic import BaseModel


class SourceDocument(BaseModel):
    """User-facing citation — what gets returned in the API response."""
    article_id: str
    title: str
    url: str
    score: float