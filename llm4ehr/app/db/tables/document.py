from datetime import datetime, timezone, timezone
from sqlalchemy import Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.sqlite import Base


class FetchedPaper(Base):
    """Tracks every successfully fetched paper. DOI is the unique key."""

    __tablename__ = "fetched_papers"

    doi: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    journal: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc)
    )


class QueryCursor(Base):
    """Stores the OpenAlex pagination cursor per query so each run advances."""

    __tablename__ = "query_cursors"

    query: Mapped[str] = mapped_column(Text, primary_key=True)
    next_cursor: Mapped[str] = mapped_column(Text, default="*")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )


class FetchJob(Base):
    """One record per pipeline run — gives users visibility into what happened."""

    __tablename__ = "fetch_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, default="running"
    )  # running | completed | failed
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
