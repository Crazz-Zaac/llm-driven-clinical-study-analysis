from pydantic import BaseModel, Field, field_validator
from typing import List
 
 
class PaperRequest(BaseModel):
    query: List[str] = Field(
        ...,
        description="List of search keywords — combined as AND query",
        examples=[["MIMIC-IV", "Retrospective"]],
    )
    max_results: int = Field(
        10,
        ge=1,
        le=25,
        description="Number of papers to fetch per run (max 25, OpenAlex page limit)",
    )
 
    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v):
        if not v or any(k.strip() == "" for k in v):
            raise ValueError("query must contain at least one non-empty keyword")
        return [k.strip() for k in v]
 
 
class PaperResponse(BaseModel):
    job_id: str
    keywords: list[str]
    cursor_used: str
    next_cursor: str | None
    total_available: int
    total_found: int
    already_fetched: int
    unavailable: int
    newly_fetched: int
    fetched_papers: list[dict]
    skipped_papers: list[dict]