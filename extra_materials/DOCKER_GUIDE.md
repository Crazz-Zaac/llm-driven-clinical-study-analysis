# Docker Setup Guide

## Overview

This docker-compose configuration provides a complete development and production-ready setup for the Clinical Study Analysis RAG system.

## Services

### 1. **Qdrant** (Vector Database)
- **Image**: `qdrant/qdrant:latest`
- **Port**: 6333 (REST), 6334 (gRPC)
- **Purpose**: Stores and retrieves document embeddings
- **Volumes**:
  - `qdrant_storage` - Persistent data storage
  - `qdrant_snapshots` - Backup snapshots
- **Health Check**: Enabled with 40s startup grace period

### 2. **API** (FastAPI Application)
- **Build**: From `docker/Dockerfile`
- **Port**: 8000
- **Purpose**: REST API for chat, ingestion, and RAG operations
- **Dependencies**: Depends on healthy Qdrant service
- **Volumes**: 
  - App code (for hot-reload during development)
  - Scripts and dataset directories
- **Environment Variables**: See `.env.example`
- **Health Check**: Enabled

### 3. **Scraper** (Optional)
- **Profile**: `scraper` - Run only when explicitly enabled
- **Purpose**: Scrape clinical literature
- **Command**: `python scripts/scrape_literatures.py`

## Environment Configuration

### Create `.env` file from template:
```bash
cp .env.example .env
```

### Required environment variables:
```bash
# Qdrant
QDRANT_URL=http://qdrant:6333          # Use service name in docker-compose
QDRANT_API_KEY=your-secure-key         # Change in production
QDRANT_COLLECTION_NAME=ehr_documents

# HuggingFace LLM
HF_API_KEY=your-huggingface-api-key    # Get from huggingface.co
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.1

# LLM Parameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
```

## Quick Start

### 1. Development Setup
```bash
# Build and start services
docker-compose up --build

# In another terminal, view logs
docker-compose logs -f api

# API will be available at http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### 2. Run Scraper
```bash
# Run scraper service (one-time)
docker-compose run --rm scraper

# Or with profile
docker-compose --profile scraper up scraper
```

### 3. Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Stop specific service
docker-compose stop api
```

## Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f qdrant

# Last 100 lines
docker-compose logs --tail=100 api
```

### Execute Commands in Container
```bash
# Interactive shell in API container
docker-compose exec api bash

# Run Python command
docker-compose exec api python -m pytest

# Check API health
docker-compose exec api curl http://localhost:8000/health
```

### Database Operations
```bash
# Check Qdrant status
docker-compose exec qdrant curl http://localhost:6333/health

# Access Qdrant with Python client
docker-compose exec api python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://qdrant:6333')
print(client.get_collections())
"
```

## Production Deployment

### Security Changes Required:

1. **Update `.env` with secure values:**
```bash
QDRANT_API_KEY=generate-a-secure-key  # Use `openssl rand -hex 32`
HF_API_KEY=your-actual-api-key
```

2. **Restrict CORS in `app/main.py`:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

3. **Disable auto-reload in `docker-compose.yml`:**
```yaml
command: uvicorn app.main:app --host 0.0.0.0 --port 8000
# Remove --reload flag
```

4. **Use production ASGI server:**
```dockerfile
# In Dockerfile, use gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

5. **Set resource limits in `docker-compose.yml`:**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

6. **Enable API authentication:**
```python
# Add Bearer token authentication to routes.py
from fastapi.security import HTTPBearer
security = HTTPBearer()

@router.post("/ingest", security_scopes=security)
async def ingest_documents(request: IngestionRequest, credentials = Depends(security)):
    # Validate token
    pass
```

## Networking

All services communicate through `clinical-rag-network` bridge network:
- **API → Qdrant**: Use service name `http://qdrant:6333` (not localhost)
- **External → API**: Use `http://localhost:8000`
- **External → Qdrant**: Use `http://localhost:6333`

## Troubleshooting

### API can't connect to Qdrant
```bash
# Check network connectivity
docker-compose exec api ping qdrant

# Check service is healthy
docker-compose logs qdrant | grep health
```

### Port already in use
```bash
# Find what's using the port
lsof -i :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # External:Container
```

### Permission denied errors
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./dataset ./scripts

# Or run docker commands with sudo (not recommended)
```

### Out of memory
```bash
# Check resource usage
docker stats

# Limit in docker-compose.yml (see Production section)
```

## File Structure Expected

```
project/
├── docker/
│   ├── Dockerfile          # Application image
│   └── docker-compose.yml  # Service orchestration
├── app/
│   ├── main.py             # FastAPI application
│   ├── api/
│   ├── rag/
│   ├── schemas/
│   └── core/
├── scripts/                # Scraper and utilities
├── dataset/                # Data storage
├── pyproject.toml          # Python dependencies
└── .env                    # Environment configuration
```

## Next Steps

1. ✅ Configure `.env` with your credentials
2. ✅ Build images: `docker-compose build`
3. ✅ Start services: `docker-compose up`
4. ✅ Test API: `curl http://localhost:8000/health`
5. ✅ Access docs: http://localhost:8000/docs
6. ✅ Ingest documents: `POST /api/v1/ingest`
7. ✅ Query documents: `POST /api/v1/rag`

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Qdrant Docker Guide](https://qdrant.tech/documentation/guides/docker/)
