# Quick Reference Checklist

## Phase 1: Component Testing ✅

### Step 1: Setup (5 min)
```bash
cd llm4ehr
pip install -e ".[dev]"
```

### Step 2: Run Tests (10 min)
```bash
# All tests
pytest tests/ -v

# Individual components
pytest tests/test_embedder.py -v              # Embedding tests
pytest tests/test_query_service.py -v         # Retrieval tests
pytest tests/test_ingestion_service.py -v     # Ingestion tests
```

### Step 3: Check Coverage (5 min)
```bash
pytest --cov=app tests/ --cov-report=html
open htmlcov/index.html
```

### Expected Results
- ✅ All embedder tests should PASS
- ✅ All query service tests should PASS (if Qdrant mocked correctly)
- ✅ All ingestion tests should PASS (if components mocked correctly)
- ✅ Coverage should be > 80%

### If Tests Fail
| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'app'` | Run `pip install -e .` first |
| `ImportError: cannot import name 'ChatModel'` | Check imports paths in test files |
| `Connection refused` to Qdrant | Don't worry - tests mock the DB, no Qdrant needed |
| LLM tests fail | That's OK for now - we'll test LLM in Phase 2 |

---

## Phase 2: Integration Testing 📊

### Create tests/test_rag_pipeline.py
```python
# Template (you fill in details):
class TestRAGPipeline:
    def test_ingest_and_retrieve(self):
        # 1. Ingest a document
        # 2. Query for it
        # 3. Verify it's retrieved
        
    def test_chat_with_context(self):
        # 1. Ingest a document
        # 2. Ask a question related to it
        # 3. Verify LLM uses the document in response
```

---

## Phase 3: API Endpoints 🔌

### Create app/api/v1/endpoints/routes.py
```python
from fastapi import APIRouter
from app.rag.ingestion.ingestion import IngestionService
from app.rag.pipeline import RAGPipeline

router = APIRouter()

@router.post("/ingest")
async def ingest(documents: list[str]):
    service = IngestionService()
    # Call service and return result
    
@router.post("/chat")
async def chat(messages: list[dict]):
    pipeline = RAGPipeline()
    # Call pipeline and return response
```

### Create app/main.py
```python
from fastapi import FastAPI
from app.api.v1.endpoints.routes import router

app = FastAPI(title="Clinical Study Analysis")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Test API
```bash
# Start server
uvicorn app.main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Clinical study about disease X"]}'
```

---

## Phase 4: Frontend 🎨

### Option A: Simple HTML (minimal)
- Single `index.html` with chat interface
- Call API from browser JavaScript

### Option B: React (recommended if you want modern UI)
```bash
npm create vite@latest frontend -- --template react
cd frontend && npm install
# Build a chat interface calling your API
```

---

## Phase 5: Docker 🐳

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e .
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

### docker-compose.yml
```yaml
version: '3'
services:
  app:
    build: .
    ports:
      - "8000:8000"
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

---

## Environment Variables to Set

### .env file
```
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=clinical_documents

# HuggingFace LLM
HF_API_KEY=your_hf_token
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.1
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Success Metrics

### By End of Phase 1
- [ ] All unit tests passing
- [ ] Code coverage > 80%
- [ ] No critical bugs in components

### By End of Phase 2
- [ ] Full RAG pipeline tested end-to-end
- [ ] Can ingest documents
- [ ] Can retrieve documents
- [ ] Can generate chat responses

### By End of Phase 3
- [ ] `/api/v1/ingest` endpoint working
- [ ] `/api/v1/chat` endpoint working
- [ ] Can test via curl or Postman

### By End of Phase 4
- [ ] User-friendly chat UI
- [ ] Document upload interface

### By End of Phase 5
- [ ] Docker image builds
- [ ] Application runs in container
- [ ] Ready for deployment

---

## Useful Commands

```bash
# Testing
pytest                                    # Run all tests
pytest -v                                # Verbose output
pytest -k embedder                       # Run tests matching pattern
pytest --cov=app                         # Coverage report
pytest --cov=app --cov-report=html      # HTML coverage report

# Development
python -m pip install -e ".[dev]"        # Install in dev mode
pip install -e .                         # Install core dependencies
python -m black .                        # Format code
python -m pytest tests/ --maxfail=1      # Stop on first failure

# API
uvicorn app.main:app --reload            # Start dev server
uvicorn app.main:app --host 0.0.0.0      # Listen on all interfaces

# Docker
docker build -t clinical-rag .           # Build image
docker run -p 8000:8000 clinical-rag     # Run container
docker-compose up                        # Start with Qdrant
```

---

## Key Files Modified/Created

- ✅ `tests/test_embedder.py` - Embedding unit tests
- ✅ `tests/test_query_service.py` - Retrieval unit tests
- ✅ `tests/test_ingestion_service.py` - Ingestion unit tests
- ✅ `pyproject.toml` - Added dev dependencies & pytest config
- ✅ `TESTING_STRATEGY.md` - Detailed testing guide
- ✅ `DEVELOPMENT_ROADMAP.md` - Phase-by-phase plan
- ✅ `QUICK_REFERENCE.md` - This file

---

## Start Here 🚀

```bash
cd llm4ehr
pip install -e ".[dev]"
pytest tests/ -v
```

Then read `TESTING_STRATEGY.md` for detailed guidance on each phase.

Good luck! 💪
