# LLM4EHR: Structured Extraction of Retrospective MIMIC-IV Studies

LLM4EHR is a local retrieval-augmented generation (RAG) system for extracting
structured study characteristics (cohort, outcome, features, preprocessing, model,
evaluation, …) from published retrospective studies that use the
[MIMIC-IV](https://physionet.org/content/mimiciv/) database. It was developed for the
Master's thesis *Structured Extraction of Retrospective MIMIC-IV Studies*
(Rabin B K, Friedrich-Alexander-Universität Erlangen-Nürnberg, 2026).

The system

* discovers open-access articles through OpenAlex and extracts their abstract,
  methods, results, discussion and conclusion from the publisher pages,
* indexes section-aware chunks in Qdrant with dense (Ollama embeddings) and sparse
  (BM25) vectors,
* answers questions with hybrid search, cross-encoder reranking and a locally
  served Ollama model, either per paper, over the most relevant papers, or for
  every paper at once (one table row per paper), and
* is evaluated against a 38-question benchmark with RAGAS and a manual review.

No patient data is processed: the input is published article text only.

```
OpenAlex ──► publisher page ──► section JSON ──► chunks ──► Qdrant (dense + BM25)
                                                               │
 question ──► hybrid search (RRF) ──► cross-encoder rerank ──► Ollama LLM ──► answer
```

## Repository layout

```
llm4ehr/
├── app/                      FastAPI backend
│   ├── main.py               application entry point (routers, /health, /docs)
│   ├── api/v1/endpoints/     REST endpoints (fetch, index, retrieve, chat, models, jobs)
│   ├── core/config.py        settings read from app/.env
│   ├── db/                   Qdrant client, SQLite (fetch jobs, active models)
│   ├── fetcher/              publisher-page text and table extraction
│   ├── research/             OpenAlex discovery and the paper-fetching workflow
│   ├── rag/
│   │   ├── chunking/         section-aware splitter (1,000 characters, 200 overlap)
│   │   ├── embeddings/       dense (Ollama) and sparse (fastembed BM25) embeddings
│   │   ├── reranking/        cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   │   ├── services/         fetch, indexing, retrieval, chat and Ollama services
│   │   ├── llm/              Ollama and hosted (OpenAI/Mistral/…) chat models
│   │   ├── prompts/          system prompts for the three retrieval scopes
│   │   ├── pipeline.py       local RAG pipeline (Ollama)
│   │   └── pipeline_remote.py RAG pipeline with a hosted model
│   └── schemas/              Pydantic request/response models
├── frontend/                 Next.js chat interface and model settings
├── docker/                   Dockerfiles, docker-compose.yml, nginx and Slurm/Apptainer files
├── rag_evaluation/           evaluation scripts and the thesis analysis notebook
├── tests/                    unit tests
├── pyproject.toml / poetry.lock / requirements.txt
└── .env.example              template for app/.env
```

## Getting started

### Requirements

* Docker with Docker Compose. An NVIDIA GPU is used by the Ollama container
  (`docker-compose.yml` requests `nvidia.com/gpu=all` through CDI; remove the
  `devices` entry to run on CPU).
* For local development and the evaluation: Python 3.11–3.12 and
  [Poetry](https://python-poetry.org/), Node.js 20 for the frontend.

### 1. Configure

```bash
cd llm4ehr
cp .env.example app/.env      # then fill in the values you need
```

`OPENAI_API_KEY` / `MISTRALAI_API_KEY` are only needed for hosted models and for
using them as RAGAS judges.

### 2. Start the stack

```bash
cd llm4ehr/docker
docker compose up -d --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API (Swagger UI) | http://localhost:8001/docs |
| Qdrant | http://localhost:6333/dashboard |
| Ollama | http://localhost:11434 |

### 3. Pull and activate models

```bash
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b

API=http://localhost:8001/api/v1
curl -X POST $API/model/activate -H 'Content-Type: application/json' \
     -d '{"model_name": "nomic-embed-text:latest", "model_type": "embedding"}'
curl -X POST $API/model/activate -H 'Content-Type: application/json' \
     -d '{"model_name": "llama3.2:3b", "model_type": "chat"}'
```

Models can also be pulled and activated from the settings page of the frontend.

### 4. Fetch and index articles

```bash
# Discover articles on OpenAlex and save one JSON record per article
curl -X POST $API/fetch/from-openalex -H 'Content-Type: application/json' \
     -d '{"query": ["MIMIC-IV", "retrospective", "machine learning"], "max_results": 10}'

# Or fetch known article URLs
curl -X POST $API/fetch/batch -H 'Content-Type: application/json' \
     -d '{"urls": ["https://link.springer.com/article/10.1186/s12873-022-00582-z"]}'

# Chunk, embed and index all fetched articles
curl -X POST $API/index/all
```

Fetched articles are stored as JSON in the backend volume
(`/app/data/fetched_articles`), with the fields `article_id`, `url`, `title`,
`abstract`, `methods`, `results`, `discussion` and `conclusion`.

### 5. Ask questions

Use the chat interface, or call the API directly:

```bash
curl -X POST $API/rag/local -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What was the size of the study cohort?"}],
  "retrieval_scope": "all_papers",
  "include_contexts": true,
  "seed": 42
}'
```

| `retrieval_scope` | Behaviour |
|---|---|
| `specific_paper` | answers from one paper (`article_id`) |
| `relevant_papers` | answers from the top-ranked papers |
| `all_papers` | answers once per indexed paper; the response is a Markdown table *Paper / Answer / Evidence* |

`temperature`, `top_p`, `max_tokens` and `seed` are passed to Ollama. Without a
`seed` the generation is sampled and repeated runs can differ.

## API overview

All routes are prefixed with `/api/v1`; the full schema is at `/docs`.

| Area | Endpoints |
|---|---|
| Fetching | `POST /fetch/from-openalex`, `POST /fetch/batch`, `GET /fetch/list`, `DELETE /fetch/articles`, `POST /fetch/reset-cursor` |
| Indexing | `POST /index/all`, `POST /index/articles` (selected `article_ids`), `POST /index/stop`, `DELETE /index` |
| Retrieval | `POST /retrieve` (chunks and merged documents without generation) |
| Chat | `POST /rag/local` (Ollama), `POST /rag/remote` (hosted provider, request-scoped API key) |
| Models | `POST /model/pull`, `POST /model/pull-stream`, `POST /model/activate`, `GET /model/list`, `GET /model/active-models`, `DELETE /model/delete` |
| Jobs | `GET /fetch/jobs`, `POST /jobs/{job_id}` |

## Evaluation (`llm4ehr/rag_evaluation`)

| File | Purpose |
|---|---|
| `evaluate_rag.py` | Queries the running API with every benchmark question, splits all-papers answers into one sample per paper, labels each answer as reported / not reported, and scores the reported cells with RAGAS (faithfulness, answer relevancy, context recall, context precision, answer correctness). |
| `rescore_saved_reference.py` | Corrects one reference answer in a saved results file and re-scores only that cell with the recorded judge (preview by default, `--apply` writes a backup and the update). |
| `review_tool.html` | Browser tool used for the manual review: labels each cell as correct, retrieval miss or generation miss and exports the CSV. |
| `thesis_results.ipynb` | Reproduces every table and figure of the thesis Results chapter from the reviewed results. |

Install the evaluation dependencies and run an evaluation while the stack is up:

```bash
cd llm4ehr
poetry install --with evaluation,dev
cd rag_evaluation
python evaluate_rag.py \
    --dataset datasets/ML4EHR.csv \
    --response-model llama3.2:3b \
    --judge-model gpt-5.6-luna \
    --generator-seed 42 \
    --api-timeout 0 --judge-timeout 1200 \
    --output results/llama_3_2b.csv
```

`python evaluate_rag.py --help` lists all options (judge provider and seed,
retrieval scope, limits, correctness threshold, …). The judge provider is chosen
from the model name: tags with `:` use Ollama, `ministral-…`/`mistral-…` use
Mistral, anything else uses OpenAI.

### Data

The benchmark, the generated and manually reviewed answers, the analysis tables,
the article corpus and the run configuration are published on Zenodo:
**https://doi.org/10.5281/zenodo.23806217**. They are not stored in this
repository. To rerun the notebook, place the files as follows:

| Zenodo | Repository |
|---|---|
| `benchmark/ML4EHR.csv` | `llm4ehr/rag_evaluation/datasets/ML4EHR.csv` |
| `results/V1/*.csv`, `results/V2/*.csv` | `llm4ehr/rag_evaluation/manually_reviewed/results_V1/`, `…/results_V2/` |

`thesis_results.ipynb` writes its figures and tables to
`rag_evaluation/analysis_outputs/thesis/`.

## Tests

```bash
cd llm4ehr
poetry install --with evaluation
poetry run python -m unittest tests.test_evaluate_rag_status tests.test_rescore_saved_reference
poetry install --with evaluation --extras dev && poetry run pytest   # full suite
```

## Running on an HPC cluster

`docker/run-containers.sbatch` starts Qdrant, the backend and the frontend from
Apptainer images (`*.sif`) in a Slurm job. Adjust `CONTAINER_DIR` and `WORK_DIR`
to your cluster before submitting.

## Citation

If you use this code or the data, please cite the thesis and the dataset:

> Rabin B K (2026). *LLM4EHR: Processed research article corpus and evaluation data
> for RAG-based extraction of MIMIC-IV studies* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.23806217
