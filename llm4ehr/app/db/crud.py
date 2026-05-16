from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db.tables.document import FetchedPaper, QueryCursor, FetchJob


#  FetchedPaper
def is_doi_fetched(db: Session, doi: str) -> bool:
    return db.query(FetchedPaper).filter_by(doi=doi).first() is not None


def mark_paper_fetched(
    db: Session, doi: str, title: str, url: str, journal: str | None = None
):
    paper = FetchedPaper(doi=doi, title=title, url=url, journal=journal)
    db.merge(paper)  # merge = insert or update if already exists
    db.commit()


def get_all_fetched(db: Session) -> list[FetchedPaper]:
    return db.query(FetchedPaper).order_by(FetchedPaper.fetched_at.desc()).all()


#  QueryCursor
def get_cursor(db: Session, query: str) -> str:
    """Returns the current cursor for a query. Defaults to '*' (first page)."""
    row = db.query(QueryCursor).filter_by(query=query).first()
    return row.next_cursor if row else "*"


def save_cursor(db: Session, query: str, next_cursor: str):
    row = QueryCursor(
        query=query, next_cursor=next_cursor, updated_at=datetime.now(timezone.utc)
    )
    db.merge(row)
    db.commit()


def reset_cursor(db: Session, query: str):
    """Reset pagination for a query back to the first page."""
    save_cursor(db, query, "*")


#  FetchJob
def create_job(db: Session, job_id: str, query: str) -> FetchJob:
    job = FetchJob(job_id=job_id, query=query, status="running")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def complete_job(db: Session, job_id: str, summary: dict):
    job = db.query(FetchJob).filter_by(job_id=job_id).first()
    if job:
        job.status = "completed"
        job.summary = summary
        job.completed_at = datetime.now(timezone.utc)
        db.commit()


def fail_job(db: Session, job_id: str, reason: str):
    job = db.query(FetchJob).filter_by(job_id=job_id).first()
    if job:
        job.status = "failed"
        job.summary = {"error": reason}
        job.completed_at = datetime.now(timezone.utc)
        db.commit()


def get_job(db: Session, job_id: str) -> FetchJob | None:
    return db.query(FetchJob).filter_by(job_id=job_id).first()


def get_recent_jobs(db: Session, limit: int = 10) -> list[FetchJob]:
    return db.query(FetchJob).order_by(FetchJob.created_at.desc()).limit(limit).all()
