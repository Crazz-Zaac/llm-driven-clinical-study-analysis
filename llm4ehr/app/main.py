"""
FastAPI application entry point for LL4EHR RAG system
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import routes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LL4EHR RAG",
    description="RAG-based system for analyzing clinical studies using LLMs",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
    return {
        "message": "LL4EHR RAG API",
        "version": "0.1.0",
        "docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "clinical-rag-api"}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("LL4EHR RAG API starting up...")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("LL4EHR RAG API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
