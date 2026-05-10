# Necessary commands
- Activating environment (locally) - `nix-shell` 
- Activating environment (on hpc): `source $WORK/poetry-env/bin/activate`
- Generating a hex code (as secret key): `openssl rand -hex 32`
- Building docker image from root folder (llm4ehr): `docker build -f docker/Dockerfile -t llm4ehr-app .`

---

# Project History

## 2026-02-25

- project setup
  - Installed necessary packages like: `langchain, beautifulsoup, pyeuropepmc, pdfminer`
  - Search for `query="MIMIC-IV AND Retrospective"` in the Europe PMC
    ```python
      results = client.search(
          query="MIMIC-IV AND Retrospective",     # query to find relevant articles
          resultType="core",
          format="json",
          abstractText=True,  # Include abstracts in results
          pageSize=100,       # Fetches 100 results per page
      )
    ```
  - Extracted the following data in JSON format
    ```python
    output_json = {
        "article_id": ,
        "title": ,
        "abstract": ,
        "doi": ,
        "journal": ,
        "publication_year": ,
        "full_text": ,
        "preprocessed": {   # data after preprocessing
            "cleaned_text": ,
            "sections": ,
            "chunks": ,
        },
    }
    ```
  - Using `RecursiveCharacterTextSplitter` from `langchain_text_splitters` to chunk the text
    - Chunking with a default size to `4000` and an overlap of `200`
    ```python
      SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=max_chars,
          chunk_overlap=overlap,
          separators=SEPARATORS,
          keep_separator=True,
      )
    ```
  - A final `index.json` file contain the following entries
    ```python
    {
        "article_id": ,
        "title": ,
        "abstract": ,
        "doi": ,
        "journal": ,
        "publication_year": ,
        "num_sections": ,
        "num_chunks": ,
    }
    ```
---

## 2026-03-01

- The issue was that `pdfminer` package just reads data _left-to-right, top-to-bottom_. It was totally ignoring the layout of the paper. For two-column layout papers, it was producing garbled results. 
  - Solution: 
    - Installed [`pymupdf4llm`](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) package as it supports multi-column pages.
- Next issue was that respective section's, for example, `Results` data wasn't being read even though the paper had this section. This was because the regex wasn't strong enough.

---

## 2026-03-16

- Created `scrape_literatures.py`
  - `scrape_articles()`: queries nature's journal and returns search results containing paper's url
  - `extract_article_html()`: extracts the clean structured form of HTML using _Beautifulsoup_
  - `extract_sections`: extracts useful sections of the paper defined in the `section_patterns` dict
    - iterates in the order of `["h1", "h2", "h3", "p", "ul", "ol", "table"]`
    - this assures that smaller headings belongs to the recent biggest heading
      - For example if the _Methods_ with recent heading is followed by sub-headings, those will be included in the Methods section as well
    - returns a dictionary of sections
  - `extract_article_id()`: extracts article id from url 
  - `save_article()`: saves a JSONL file appending all the incoming literature contents
  - `process_article()`: main method that calls all other methods
- After running the `scrape_literatures.py`, a file `scraped_articles/scraped_articles.jsonl` will be created 

---

## 2026-03-22

- Restructured the project for dockerizing into services

---

## 2026-03-23

- Setting up project in HPC 
  - creating virtual environment
  - installing packages

- Starting with embedding models
  - decided to use `qdrant`
    - because additional configurations for storing metadat isn't necessary unlike FAISS
    - provides APIs for insert/search/update
    - runs as a service
    - FAISS is more of a manual configurations
    
  - decided to use SOTA `sentence-transformers`

- Basic workflow
  - Chunk data -> Embed it -> store in Qdrant [indexing + storage]

--- 

## 2026-03-29 (Minimal RAG pipeline)
### Indexing Phase (First Run)
```
📄 Documents                             → Load raw data (PDFs, text, etc.)
      ↓
✂️ Chunking (LangChain)                  → Split text into smaller meaningful pieces
      ↓
🔢 Embeddings (Sentence-Transformers)    → Convert each chunk into a vector
      ↓
🗄️ Qdrant (store + index)                → Store vectors + build search index
```

### Chat/Interaction Phase
```
💬 User Query                → User asks a question
      ↓
🔢 Embedding                 → Convert query into a vector
      ↓
🔍 Qdrant Search (top k)     → Find similar vectors (Top-K chunks)
      ↓
📚 Context                   → Collect retrieved chunks
      ↓
📝 Prompt Assembly           → Combine context + question
      ↓
🤖 LLM                       → Generate answer
      ↓
💡 Response                  → Return final answer to user
```
- Created `schemas/ingestion_schema.py `
  - Ingestion request will have document
  - Ingestion response will receive a sucess message with metadata

- Created `ingestion/ingestion.py` 
  - Chunks the document
  - Embeds each chunk
  - For each chunk:
    - creates a metadata
    - adds it to the batch vector
  - Upserts each vectors of batch size of 100

- Created `schemas/chat_schema.py`
  - Created 3 chat roles: user, assistant and system
  - The chat messages contains role and the content
  - During chat request, only messages is passed
  - The chat response will receive response and the documents

- Created `llm/chat_model.py`
  - call hugging face model
  - formats the message for prompt based on roles
  - then it generates the response

- Created `services/chat_service.py`
  - initiates chat model
  - generate chat response

- Created `schema/query_schema.py`
  - It takes query as a request
  - It returns response, source_documents 

- Created `retrieval/query.py`
  - it embeds the request
  - searches the vector db
  - extracts the texts of retrieved documents
  - Returns response and the source documents

---

## 2026-03-30

- Restructured the project
  - removed `ingestion/ingestion.py` and instead created `services/ingestion_service.py`
  - removed `retrieval/query.py` and instead created `services/retrieval_service.p`

- Created tests for testing each rag components
```
tests/
├── test_embedder.py              ✅ Component: Text embeddings
├── test_ingestion_service.py     ✅ Service: Document ingestion 
├── test_query_service.py         ✅ Service: Document retrieval 
├── test_chat_service.py          ✅ Service: Chat handling 
└── test_rag_pipeline.py          ✅ Pipeline: End-to-end RAG
```
- The tests include:
  - Initialization
  - Successful retrieval
  - No results handling
  - Error handling
  - Embedding validation
  - Custom top_k parameter

- Running the Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ingestion_service.py -v
pytest tests/test_rag_pipeline.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test class
pytest tests/test_rag_pipeline.py::TestRAGPipeline -v

# Run specific test method
pytest tests/test_rag_pipeline.py::TestRAGPipeline::test_pipeline_run_with_documents -v
```

---

## 2026-04-03

- Updated `docker-compose.yml` 
- Updated `Dockerfile`
  - resolved the issue with poetry installation and package installation
  - resolved the issue with directory structure
- Successfully built docker image

---

## 2026-04-04

- Created schema, services and edpoint for scraping paper from nature journal
  - Created `schema/scrape_schema.py`
  - Created `services/scrape_service.py`
  - Created `api/v1/scrape` endpoint 
- Saves the scrapped documents to `app/data/` in the container

---

## 2026-04-05

- created frontend container
- generated frontend in next js using [v0](v0.app)
- adjusted the UI/UX design to include the following functionalities:
  - allow user to download and use ollama embedding models 
  - allow user to use hosted llm models by taking the API keys
  - allow user to download ollama chat models and vision models
  - allow user to upload PDFs (mulitple uploads) or scrape from nature journal
  - allow user to change model parameter (temperature)
  
---

## 2026-04-06

- Created `.sif` and `.def` files for singularity containerization
  - `apptainer build backend.sif backend.def`
  - `apptainer build frontend.sif frontend.def`
  - `apptainer build qdrant.sif qdrant.def`
- Copied the singularity images to HPC and ran the containers
  - `scp backend.sif user@hpc:/path/to/destination`
- Successfully built singularity image and ran the container in HPC


---

## 2026-04-30

- Adjusted the `tests/embedding_test.py` to test the embedding of the abstract of the paper
  - `PYTHONPATH=. python tests/embedding_test.py`
  - Qdrant is running: `http://localhost:6333/dashboard#/collections`
- Created `tests/query_test.py` to test the retrieval of the relevant documents from the vector database
  - `python -s tests/query_test.py`
  - The retrieval worked and the results were relevant to the query

---

## 2026-05-02
- Updated `llm/chat_model.py`
  - Initially the `generate_response` method was only taking the question as input and generating a response without considering the retrieved documents. This was not ideal for a RAG pipeline where the LLM should generate responses based on the retrieved context.
  - Updated the `generate_response` method to take both the question and the retrieved documents as input. The retrieved documents are now included in the prompt that is sent to the LLM, allowing it to generate more informed and accurate responses based on the provided context.
-  Adjusted the `qdrant_client.py`
   -  Initially the client response was being returned as a list of dictionaries. This was causing issues in the retrieval service when trying to access the document attributes.
   -  Updated to return points with payload as a list of `Document` objects instead of dictionaries. This allows for easier access to document attributes like `title`, `abstract`, etc. in the retrieval service.
- Updated the `schemas/query_schema.py`
  - Updated the `QueryResponse` schema to have `source_documents` as a list of `RetrievedDocument` objects instead of a list of dictionaries. 
  - The `RetrievedDocument` schema contains fields like `title`, `abstract`, `article_id`, etc. which are relevant for the retrieved documents.
- Updated the `retrieval_service.py`
  - In the retrieval service, after getting the search results from Qdrant, I now create a list of `Document` objects from the retrieved points. Each `Document` object contains the relevant metadata and content from the retrieved chunks.
- Updated `schemas/chat_schema.py`
  - Updated the `ChatResponse` schema to have `source_documents` as a list of dictionaries instead of a list of strings. Each dictionary represents a retrieved document with its metadata and content. 
-  Updated the `pipeline.py`
   - In the pipeline, when assembling the retrieved documents for the prompt, I now access the document attributes directly from the `Document` objects instead of trying to access them as dictionary keys. This allows for a more structured and clear way to format the retrieved documents in the prompt.
- Updated `system_prompt.py`


---

## 2026-05-08

- Updated the `services/ingestion_service.py`
  - In the ingestion service, I updated the method that processes the document to also include the `conclusion` section in addition to the `abstract`, `methods`, and `results`. 
  - I was previously looping chunk by chunk and creating the metadata which was not efficient and slower. Now I make a list of all chunks, embed them together then loop throug the chunk and embeddings together to create the metadata and the point for upserting. This is more efficient as it reduces the number of calls to the embedding model and allows for batch processing of chunks.
- Created the `services/indexing_service.py`
  - created methods to index the documents into the vector database
  - created methods to delete the collection and the indexed files
  - created methods to save the embeddings in the local directory as JSON files
- Updated the `schemas/indexing_schema.py`  
  - Updated the `DocumentSchema` to include a new field for `conclusion`
- Updated the `scrapper/article_scrapper.py`
  - Previously I hadn't extracted the title of the article. Now I am extracting the `title` of the article and including it in the scrapped contents  
- Updated the `api/v1/endpoints/routes.py`
  - Added a new endpoint for deleting all indexed documents and the collection in the vector database. This allows for a clean slate when needed, especially during testing or when re-indexing documents. The endpoint will delete all JSON files in the embeddings directory and also delete the collection from Qdrant.
  - This will be later rewired to frontend to allow users to clear the indexed documents and start fresh when needed.
  
  ---

## 2026-05-10

- Updated `indexing_schema.py`
  - Added `IndexFromScrapedRequest` to include a list of article ids
- Updated `indexing_service.py`
  - Added a method `index_from_scraped` which takes the scrapped articles from the `data/` directory, processes them and indexes them into the vector database. This method will be called after scraping new articles to ensure that they are indexed and available for retrieval.
- Updated `scrape_service.py`
  - Added `scrap_text_batch` method which allows for scraping multiple articles in a batch. This method will loop through a list of article URLs, scrape each one and save the contents to the `data/` directory. 
  - Added `list_scraped_articles` method which lists all the scrapped articles available in the `data/` directory. This can be useful for keeping track of what articles have been scrapped and are available for indexing.
- Updated the `routes.py`
  - Added a new endpoint for batch scraping articles from a list of URLs. This endpoint will take a list of article URLs, call the `scrap_text_batch` method in the scrape service to scrape each article and save the contents. After scraping, it will call the `index_from_scraped` method in the indexing service to process and index the newly scrapped articles into the vector database. 