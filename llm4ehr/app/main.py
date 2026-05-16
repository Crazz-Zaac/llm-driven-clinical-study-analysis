"""
FastAPI application entry point for LLM4EHR RAG system
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import routes
from app.db.sqlite import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("LLM4EHR RAG API starting up...")
    init_db()  # ← creates SQLite tables if they don't exist yet
    logger.info("SQLite database initialized.")
    yield
    # Shutdown
    logger.info("LLM4EHR RAG API shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="LLM4EHR RAG",
    description="RAG-based system for analyzing clinical studies using LLMs",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router, prefix="/api/v1", tags=["v1"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LLM4EHR RAG API", "version": "0.1.0", "docs": "/docs"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "llm4ehr-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
