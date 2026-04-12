# Testing Strategy & Roadmap

## Current Status
Your RAG pipeline components are in place:
- ✅ TextEmbedder (embedding)
- ✅ QueryService (retrieval)
- ✅ IngestionService (data insertion with batching)
- ✅ ChatModel (LLM integration)
- ✅ Config & centralized settings

## Recommended Path Forward: **Test First, Then Build APIs**

### Phase 1: Unit Tests (Component Testing) ✅ 
**Goal**: Verify each component works correctly in isolation

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_embedder.py -v

# Run with coverage
pytest --cov=app --cov-report=html
```

**Tests included:**
1. `test_embedder.py` - Vector embedding dimensions, semantic similarity
2. `test_query_service.py` - Document retrieval, error handling
3. `test_ingestion_service.py` - Batch insertion, metadata preservation

### Phase 2: Integration Tests (Component Interaction)
**Goal**: Test components working together

```bash
# Create tests/test_rag_pipeline.py
pytest tests/test_rag_pipeline.py
```

**What to test:**
- Documents ingest → vectors stored
- Query → retrieves relevant docs
- Chat → generates response using retrieved docs

### Phase 3: API Endpoints
**Goal**: Create REST endpoints

```python
# app/api/v1/endpoints/routes.py
from fastapi import APIRouter, HTTPException
from app.rag.ingestion.ingestion import IngestionService
from app.rag.pipeline import RAGPipeline

router = APIRouter()

@router.post("/ingest")
async def ingest_documents(documents: list[str]):
    """Ingest documents into the vector database"""
    service = IngestionService()
    result = service.ingest_documents({"documents": documents})
    return result

@router.post("/chat")
async def chat(messages: list[dict]):
    """Chat endpoint with RAG"""
    pipeline = RAGPipeline()
    result = pipeline.run(ChatRequest(messages=messages))
    return result
```

### Phase 4: Frontend (Optional but recommended)
- Simple UI for testing chat
- Document upload interface

### Phase 5: Docker & Deployment
- Build containerized application
- Deploy to production

---

## Why This Order?

| Order | Approach | Pros | Cons |
|-------|----------|------|------|
| **1. Test Components First** ✅ | Unit → Integration → API → Frontend → Docker | Early problem detection, faster debugging, parallel work | Takes more time upfront |
| | **2. Build Everything at Once** ❌ | Faster initial development | Hard to debug, all components fail together |

---

## Quick Start Commands

```bash
# 1. Install dev dependencies
cd llm4ehr
pip install -e ".[dev]"

# 2. Run unit tests
pytest tests/ -v

# 3. Check coverage
pytest --cov=app tests/

# 4. Run specific component test
pytest tests/test_embedder.py -v

# 5. Run with markers
pytest -m unit
```

## Next Steps

1. **Run the unit tests** - This will show you if embeddings, retrieval, and ingestion work
2. **Fix any failing tests** - Debug component issues one at a time
3. **Create integration tests** - Test the full RAG pipeline end-to-end
4. **Build API endpoints** - Once components are verified
5. **Add frontend** - Once APIs are working
6. **Containerize** - Final step before deployment

---

## Test Coverage Goals

- Unit tests: **80%+ coverage** of core RAG logic
- Integration tests: Critical path scenarios
- API tests: Happy path + error cases

Use `pytest --cov=app --cov-report=html` to visualize coverage.
