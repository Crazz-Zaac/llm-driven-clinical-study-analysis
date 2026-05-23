from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import logging

from app.db import crud
from app.db.sqlite import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Jobs"])


# Crawler Job Endpoints
@router.get("/fetch/jobs")
async def list_jobs(limit: int = 10, db: Session = Depends(get_db)):
    """List recent fetch jobs and their statuses."""
    try:
        jobs = crud.get_recent_jobs(db, limit=limit)
        return {"jobs": [job.model_dump() for job in jobs]}
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List jobs error: {str(e)}",
        )


@router.post("/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status and summary of a specific fetch job."""
    try:
        job = crud.get_job(db, job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return {"job": job.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get job status error: {str(e)}",
        )
