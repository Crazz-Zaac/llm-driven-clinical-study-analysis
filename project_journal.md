# Necessary commands
- Activating environment (locally) - `nix-shell` 
- Activating environment (on hpc): `source $WORK/poetry-env/bin/activate`
- Generating a hex code (as secret key): `openssl rand -hex 32`
- Building docker image from root folder (llm4ehr): `docker build -f docker/Dockerfile -t llm4ehr-app .`
- Generating **requirements.txt** from poetry: `poetry export -f requirements.txt --output requirements.txt --without-hashes`
- Add package to poetry only for evaluation (for example): `poetry add -G evaluation streamlit`
- Testing scripts: `PYTHONPATH=. python tests/embedding_test.py`
- Running RAGAs evaluation script: `python evaluate_rag.py --api-timeout 0 --judge-timeout 1200 --dataset datasets/ML4EHR.json --retrieval-scope all_papers --judge-model qwen2.5:7b --output results_v3/llama_3_2_1b__qwen2_5_7b.csv`
---

# Project History

## 2026-02-25: Basic project setup and initial data extraction

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

## 2026-03-01: PDF Scraping and Section Extraction

- The issue was that `pdfminer` package just reads data _left-to-right, top-to-bottom_. It was totally ignoring the layout of the paper. For two-column layout papers, it was producing garbled results. 
  - Solution: 
    - Installed [`pymupdf4llm`](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) package as it supports multi-column pages.
- Next issue was that respective section's, for example, `Results` data wasn't being read even though the paper had this section. This was because the regex wasn't strong enough.

---

## 2026-03-16: Paper Scraping and Section Extraction

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

## 2026-03-22: Project Restructuring

- Restructured the project for dockerizing into services

---

## 2026-03-23: HPC Setup and Initial Embedding Models

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

## 2026-03-29: Minimal RAG pipeline
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

## 2026-03-30: Project Restructuring

- Restructured the project
  - removed `ingestion/ingestion.py` and instead created `services/ingestion_service.py`
  - removed `retrieval/query.py` and instead created `services/retrieval_service.py`
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

## 2026-04-03: Docker Setup and Configuration

- Updated `docker-compose.yml` 
- Updated `Dockerfile`
  - resolved the issue with poetry installation and package installation
  - resolved the issue with directory structure
- Successfully built docker image

---

## 2026-04-04: Paper Scraping and Indexing

- Created schema, services and edpoint for scraping paper from nature journal
  - Created `schema/scrape_schema.py`
  - Created `services/scrape_service.py`
  - Created `api/v1/scrape` endpoint 
- Saves the scrapped documents to `app/data/` in the container

---

## 2026-04-05: Frontend Setup and Integration

- created frontend container
- generated frontend in next js using [v0](v0.app)
- adjusted the UI/UX design to include the following functionalities:
  - allow user to download and use ollama embedding models 
  - allow user to use hosted llm models by taking the API keys
  - allow user to download ollama chat models and vision models
  - allow user to upload PDFs (mulitple uploads) or scrape from nature journal
  - allow user to change model parameter (temperature)
  
---

## 2026-04-06: Singularity Containerization for HPC

- Created `.sif` and `.def` files for singularity containerization
  - `apptainer build backend.sif backend.def`
  - `apptainer build frontend.sif frontend.def`
  - `apptainer build qdrant.sif qdrant.def`
- Copied the singularity images to HPC and ran the containers
  - `scp backend.sif user@hpc:/path/to/destination`
- Successfully built singularity image and ran the container in HPC


---

## 2026-04-30: Embedding and Retrieval Tests

- Adjusted the `tests/embedding_test.py` to test the embedding of the abstract of the paper
  - `PYTHONPATH=. python tests/embedding_test.py`
  - Qdrant is running: `http://localhost:6333/dashboard#/collections`
- Created `tests/query_test.py` to test the retrieval of the relevant documents from the vector database
  - `python -s tests/query_test.py`
  - The retrieval worked and the results were relevant to the query

---

## 2026-05-02: RAG Pipeline Update
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

## 2026-05-08: Endpoints and routes for indexing and deleting documents

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

## 2026-05-10: Indexing and fetching updates

- Updated `indexing_schema.py`
  - Added `IndexFromScrapedRequest` to include a list of article ids
- Updated `indexing_service.py`
  - Added a method `index_from_scraped` which takes the scrapped articles from the `data/` directory, processes them and indexes them into the vector database. This method will be called after scraping new articles to ensure that they are indexed and available for retrieval.
- Updated `scrape_service.py`
  - Added `scrap_text_batch` method which allows for scraping multiple articles in a batch. This method will loop through a list of article URLs, scrape each one and save the contents to the `data/` directory. 
  - Added `list_scraped_articles` method which lists all the scrapped articles available in the `data/` directory. This can be useful for keeping track of what articles have been scrapped and are available for indexing.
- Updated the `routes.py`
  - Added a new endpoint for batch scraping articles from a list of URLs. This endpoint will take a list of article URLs, call the `scrap_text_batch` method in the scrape service to scrape each article and save the contents. After scraping, it will call the `index_from_scraped` method in the indexing service to process and index the newly scrapped articles into the vector database. 

---

## 2026-05-15: Paper fetching workflow using OpenAlex API
- Renamed `scrape` to `fetch` in the service and the endpoint to better reflect the functionality of fetching articles from nature journal. Scraping is more of a general term for extracting data from websites, while fetching specifically refers to retrieving articles from a source like nature journal. This renaming will help clarify the purpose of the service and endpoint.
- Initially it was only possible fetch articles one by one  or in a batch by providing a list of nature's article URLs. Now I tend to implement a new workflow using `OpenAlex` API to search for articles based on keywords and then fetch and index the top results. This will allow for a more dynamic and flexible way to find relevant articles without needing to manually provide URLs. The workflow will be as follows:
  - OpenAlex returns a list of articles based on the provided keywords
  - For each article in the results, UNPPAYWALL API is used to check if the full text is available and to get the URL for fetching
  - The article is then fetched using the existing fetching methods and the contents are saved to the `data/` directory
  - After fetching, the article is processed and indexed into the vector database using the existing indexing methods
- This workflow will be implemented in the `fetch_service.py` and a new endpoint will be added to the `routes.py` to allow users to trigger this

---

## 2026-05-16: Indexing service update
- Updated the `indexing_service.py`
  - Initially, I was directly embedding combining the different sections of the paper per document without chunking them into smaller pieces. This was not ideal as it could lead to loss of important information and context in the embedding process, especially for longer papers. Now I have updated the `index_documents` to section-aware chunking. This means that each section of the paper (abstract, methods, results, conclusion) will be chunked separately and embedded separately. This way when the LLM gets context back, for example, it also knows "this is from the Methods section of paper X" which helps it reason better.

---

## 2026-05-23

- **Embedding models:**
nomic-embed-text:latest(~274 MB), mxbai-embed-large:latest (~670 MB), qwen3-embedding:0.6b (~600 MB), qwen2.5-embedding:0.6b (~600 MB), gemma3-embedding:0.6b (~600 MB), phi4-embedding:0.6b (~600 MB)

- **Chat models:**
llama3.2:3b, qwen2.5:3b (1.9GB), gemma3:1b(1.2GB), qwen3:3b (2.1GB), phi4-mini:3.8b(2.3GB), phi4:7.1b (5.1GB)

- Separated all the routes into different files based on their functionality. 
- Updated the indexing service
  - The delete collection method deleted the collection but never recreated it. Now after deleting the collection, it is immediately recreated to ensure that the collection is always available for indexing new documents. This also helps in avoiding issues where the collection might not exist when trying to index new documents after deletion.
- Added ollama schema and service to manage the embedding and chat models
  - Ollama schema: 
    - `PullModelRequest`: pulls the requested model name but requires the model type to be specified (embedding, chat, vision)
    - `PullModelResponse`: returns a success message after pulling the model
    - ActivateModelRequest: activates the requested model name and type
    - ActivateModelResponse: returns a success message after activating the model
  - Ollama service:
    - `pull_model`: takes the model name and type, calls the ollama API to pull the model and returns a success message
    - `get_embedding_dimension`: takes the model name, calls the ollama API to get the model details and returns the embedding dimension for embedding models. This is important for ensuring that the vector database is configured with the correct dimension for the embeddings being used.
    - `get_model_size_by_bytes`: takes the model name, calls the ollama API to get the model details and returns the size of the model in bytes. This is to restrict the user from pulling models that are larger than the specified `MAX_MODEL_SIZE` in the configuration. If the model size exceeds the limit, a ValueError is raised to inform the user.
- Added `active_models.py`
  - This is a sqlite table to keep track of the active embedding and chat models. It has four fields: `model_name`, `model_type`, `embedding_dimension` and `activated`. This allows us to easily manage and retrieve the currently active models for both embedding and chat functionalities. When a model is activated through the API, it updates this table to reflect the active model and its details. This way, when the embedding service or chat service needs to use the active model, it can query this table to get the necessary information about which model is currently active and its embedding dimension if it's an embedding model.
- Created new endpoints for all of the above functionalities

---

## 2026-05-28
- Create `ollama_chat_model.py` to chat with the locally currently active chat model
  
- Separated remote model into `remote_chat_model.py` which will call the Hugging Face API to generate responses using the remote models
  
- Updated `indexing_routes.py` to include `/index-all` endpoint which now indexes all documents present in the `data/fetched_articles` directory.
  - Renamed the endpoints for indexing to be more intuitive and clear.
  
- Separated chat models as remote and local
  - `remote_chat_model.py` currently calls Hugging Face API to generate responses. 
  - `ollama_chat_model.py` will call the currently active chat model in the ollama service to generate responses. 

- Updated `chat_routes.py`
  - Changed `/chat/remote` endpoint and changed it to `/rag/remote`
- Updated `query_schema.py`
  - Updated `RetrievedDocument` schema to include `combined_text` and `section` which will be used to provide more context to the LLM when generating responses. The `combined_text` will contain the combined text of the chunk along with its metadata (like title, section, etc.) which can help the LLM understand the context better and generate more accurate responses.

- Updated `pipeline.py`
  - Initially I was including the retrieved documents in both system prompt and user prompt which was redundant. Now I have updated it to only include the retrieved documents in the system prompt.
  
- Separated the RAG pipeline into `pipeline.py` for using local models and `pipeline_remote.py` (which inherits pipeline base class but initializes to use the `remote_chat_model`) for using remote models. 

- Updated `qdrant_client.py` 
  - There was an issue with collection creation. I was calling the `create_collection` method in the initialization of the client which was then calling the `recreate_collection` method which was deleting the collection immediately after creating it. 

- Updated `system_prompt.py`
  - Updated the system prompt to include instructions for the LLM to use the section information of the retrieved documents when generating responses. T

- The model response was including multiple duplicate articles in the source documents which is indeed correct because the same article can have multiple chunks that are relevant to the query as I'm chunking all the sections of the paper separately. However, to avoid confusion for the user, I now created a `source_document_schema.py` that includes only relevant information (article_id, url, title and score) for the user. For LLM's context we use `llm_docs` with all the information and metadata of the retrieved chunks.

---

## 2026-06-04: Backend API Integration

- Integrated the backend API with the frontend
  - Added functionality in the frontend to allow users to pull and activate embedding and chat models from ollama through the API
  - Added functionality in the frontend to allow users to upload PDFs or fetch articles from nature journal and then index them through the API
  - Added functionality in the frontend to allow users to ask questions and get responses using both local and remote models through the API
  - Updated the UI/UX design to accommodate the new functionalities and make it more user-friendly

---

## 2026-06-05: Retrieval and Indexing Service Updates

- Updated `retrieval_schema.py` to include `top_k` field in the `QueryRequest` which allows users to specify how many relevant chunks they want to retrieve from the vector database.
- Updated the `indexing_service.py` to include abstract as well in the payload of the point when indexing the chunks. Initially, I was unknowingly leaving out the abstract from the indexed metadata which is an important part of the paper and can provide valuable context for the LLM during retrieval.
- Updated the `retrieval_service.py` to have `max_chunks_per_article` parameter which limits the number of chunks retrieved per article. 

---

## 2026-06-06: Re-ranking Implementation

- Updated the `retrieval_service.py` to include a re-ranking step after retrieving the relevant chunks from the vector database. Initially, I was just retrieving the top k chunks based on their similarity scores from Qdrant and passing them to the LLM for generating responses. However, this approach can lead to suboptimal results as it does not take into account the relevance of the sections of the paper to the query. For example, if the query is about the results of a clinical study, then chunks from the "Results" section should be ranked higher than chunks from the "Methods" section, even if their similarity scores are similar.
- Implemented a `Reranker` class that uses a `cross-encoder/ms-marco-MiniLM-L-6-v2` model to re-rank the retrieved chunks based on their relevance to the query, taking into account the content of the chunks and their section information. I am using `cross-encoder/ms-marco-MiniLM-L-6-v2` model for re-ranking which is a smaller and efficient model suitable. The re-ranking step will help ensure that the most relevant chunks are prioritized in the context provided to the LLM, leading to more accurate and informed responses.
- Even with the implementation of the re-ranking, the model isn't able to answer the questions like:
  - Q: What ml models were trained according to the paper'Development and validation of machine learning models for predicting extubation failure in patients undergoing cardiac surgery: a retrospective study'?
  - A: The XGBoost algorithm, logistic regression (LR), and random forest (RF) models were trained.
- This response is correct but for the following question:
  - Q: Give me the final performance metric scores of all the trained models according to the paper'Development and validation of machine learning models for predicting extubation failure in patients undergoing cardiac surgery: a retrospective study'?
  - A: According to 'Development and validation of machine learning models for predicting extubation failure in patients undergoing cardiac surgery: a retrospective study' (Scientific Reports), the XGBoost algorithm had the highest performance metrics with an AUC of 0.793, Mean Precision of 0.700, and Brier Score of 0.150
- Here, the model is only able to retrieve the performance metrics for the XGBoost model but not for LR and RF. 
- Another thing I noticed is that the retrieved chunks and re-ranked chunks are almost always from the `Abstract`, `Results`, `Discussions` and `Conclusion` sections but not from the `Methods` section. This is because the `Methods` section contains more technical details and specific terminologies which might not be as semantically similar to the query as the other sections, even though it might contain relevant information about the trained models.
- Also, right now I'm only fetching 3 `max_chunks_per_article`. This might not be enough to capture all the relevant information from the `Methods` section which can lead to missing out on important details about the trained models and their performance metrics. I might need to increase this number to ensure that I'm capturing enough information from the `Methods` section as well, which can help the model provide more comprehensive answers to questions about the trained models and their performance metrics.

---

## 2026-06-12: Retrieval Service Update

- Initially I was only retrieving 20 vectors from the vector database for each query which might not be enough to capture all the relevant information, especially for longer papers with more sections and details. Now I have updated the retrieval service to retrieve up to 50 vectors per query.
- This improved the reponse of the previous queries correctly.
  - Q: Give me the final performance metric scores of all the trained models according to the paper'Development and validation of machine learning models for predicting extubation failure in patients undergoing cardiac surgery: a retrospective study'?
  - A: According to [Development and validation of machine learning models for predicting extubation failure in patients undergoing cardiac surgery: a retrospective study], the final performance metric scores of all the trained models are as follows:
    - XGBoost: AUC 0.793, Mean Precision 0.700, Brier Score 0.150
    - Logistic regression: AUC 0.766, Mean Precision 0.553, Brier Score 0.173
    - Random forest: AUC 0.791, Mean Precision 0.510, Brier Score 0.181
- I also thought that only returning top 5 re-ranked chunks might not be enough for the LLM to generate a comprehensive response, especially for complex queries that require information from multiple sections of the paper but as a downside, the inference time will increase as the LLM will have to process more chunks of information. I might need to experiment with different numbers of re-ranked chunks to find the right balance between response quality and inference time.
- To make the search faster for queries that explicitly mention the paper names, I think it would be beneficial to implement a more structured retrieval approach where if the query mentions a specific paper, we can first filter the chunks based on the article_id or title before applying the re-ranking. This way, we can ensure that we are only retrieving and re-ranking chunks from the relevant paper, which can lead to faster retrieval and more accurate responses for queries that are focused on specific papers.
- Updated the `retrieval_service.py` to include a check for paper title in the query. It extracts the title fragment from the query, search the vector database for the article_id based on the title fragment, and then retrieves chunks that are only from that specific article for re-ranking. This structured retrieval approach can help improve the relevance of the retrieved chunks and speed up the retrieval process for queries that mention specific paper titles.

---

## 2026-06-18: Indexing Service Update

- Updated the `indexing_service.py` to include the section name in the chunk text as well. Initially, I was only including the content of the chunk without explicitly mentioning which section it belongs to. I was only including the section information in the payload metadata but not in the chunk text itself. Now I have updated the chunk text to include the section name as a prefix. For example, if the chunk is from the Methods section, the chunk text will start with "Methods: " followed by the content of the chunk. This further improved the relevance of the retrieved chunks and the quality of the responses generated by the LLM, especially for queries that are focused on specific sections of the paper. By explicitly including the section name in the chunk text, it provides clearer context to the LLM about where the information is coming from, which can help it reason better and generate more accurate responses based on the relevant sections of the paper.

---

## 2026-06-20: Embedding and Hybrid Search Update

- Updated the `embedder.py`:
  - Included the SparseTextEmbedding `Qdrant/bm25` model 
  - Separated the embedding logic for dense and sparse embeddings into two different methods: `embed_sparse` and `embed_dense`. Ollama embedding models will be embedded using the `embed_dense` method while the `Qdrant/bm25` model will be embedded using the `embed_sparse` method. 
  - Also added `embed_sparse_batch` method to allow for batch embedding of documents using the sparse embedding model. This is important for efficiency when embedding a large number of documents with the sparse model.

- Updated `qdrant_client.py` to include `hybrid_search_vectors()` and `hybrid_search_vectors_filtered()` methods. 
  - `hybrid_search_vectors()`: This method performs a hybrid search using both dense and sparse embeddings. It takes the `query vector`, `sparse query vector`, and other search parameters to return relevant chunks based on a combination of dense and sparse similarity scores.
  - `hybrid_search_vectors_filtered()`: This method is specifically used when the user query mentions a specific paper title. It performs a hybrid search but also includes a filter to only retrieve chunks from the specified article. This allows for more focused retrieval when the query is about a specific paper, ensuring that the retrieved chunks are not only relevant to the query but also from the correct source.
  - Updated the `create_collection()` method to set up the collection for hybrid search by including both dense and sparse vector configurations.
- Updated the `indexing_service.py` to include both dense and sparse embeddings when indexing the chunks into the vector database. Each chunk will now have both a dense embedding (from the ollama embedding model) and a sparse embedding (from the Qdrant/bm25 model) stored in the vector database. 
- Updated the `retrieval_service.py` to utilize the new hybrid search methods in the Qdrant client. Depending on whether the query mentions a specific paper title or not, it will call either `hybrid_search_vectors()` or `hybrid_search_vectors_filtered()` to retrieve the relevant chunks based on both dense and sparse embeddings. 

---

## 2026-06-21: Query Understanding Issue
- Issues:
  - LLM doesn't understand `models` as `machine learning models` and it is retrieving chunks from the `Methods` section which contains the details of the trained machine learning models but since the query is asking about `models`, it is giving more weight to chunks that contain the word "model" in them, even if they are not from the Methods section. This is because the term "model" can be used in different contexts in a paper, and the LLM might be picking up on chunks that mention "model" without fully understanding that the user is specifically asking about machine learning models. To address this issue, I might need to implement a more sophisticated approach to understand the context of the query better and ensure that the retrieved chunks are more relevant to the specific information being asked for, rather than just relying on keyword matching.


---

## 2026-06-26: RAG Pipeline Evaluation and Updates
- Installed `ragas` package locally under `[evaluation]` in `pyproject.toml` to evaluate the RAG pipeline. This is excluded from the running docker image to keep the image size smaller. 
- Updated the `retrieval_service.py`:
  - I was not setting the `search_results` to None when the query didn't contain any paper title and instead doing a Hybrid Search with `article_id = None`. This builds a qdrant filter with something like `match: None` and return 0 results. But just right after that, I was checking if the `search_results` is None and then doing a Hybrid Search without any filter. This was causing the retrieval service to always return 0 results for queries that didn't contain any paper title. Now I have updated the logic to set `search_results` to None when the query doesn't contain any paper title and directly do a Hybrid Search without any filter in that case. This ensures that the retrieval service can return relevant results for queries that are not specific to any paper title.
  - I also increased the `max_chunks_per_article` parameter to 5 to ensure that enough relevant chunks are retrieved from each article, especially for queries that require information from multiple sections of the paper. 
  - To make the RAGAs evaluation easier, I also added a `context` field to the `QueryResponse` schema which contains the combined text of all the retrieved chunks. 
- Updated the `pipeline.py` to include the `context` field only when the `ChatRequest` has `include_context=True`. This ensures that the blobby context is exclude from the UI and only used for evaluation purposes when needed.

---
## 2026-07-03: Evaluation of RAG Pipeline

- Updated the `evaluate_rag.py` to include the `context` field in the evaluation results for allowing more comprehensive evaluation of the RAG pipeline by including the context provided to the LLM along with the generated response and source documents. The evaluation results will now include the following fields:
  - `query`: The user query
  - `response`: The response generated by the LLM
  - `source_documents`: The list of source documents retrieved from the vector database
  - `context`: The combined text of all the retrieved chunks that were provided to the LLM as context for generating the response
- Evaluated the RAG pipeline using the `ragas` package and saved the results to a CSV file for further analysis. 
  - Performed the following combination of evaluations:
    - Exp 1:
      - Chat model: `qwen2.5-3b`, Judge model: `qwen2.5-7b`
    - Exp 2:
      - Chat model: `llama3.1:8b`, Judge model: `qwen2.5-3b`
    - Exp 3:
      - Chat model: `llama3.2:1b`, Judge model: `qwen2.5-3b`

---

## 2026-07-09: Updates to Retrieval Service

- Updated the `retrieval_service.py` 
  - Initially I was:  
    - Qdrant hybrid search -> group chunks by article -> merge chunks per article -> rerank articles -> keep top 5 articles
    - This approach maybe ideal for queries that are asking about a specific paper because the re-ranker is just choosing the most relevant chunks from that known paper. But for general queries (not specific to any paper) this may exclude useful papers (as hybrid search was limiting only 5 * 10 = 50) and later I was grouping by article before reranking which may exclude relevant chunks from other papers (reranker never gets a chance to see them).
  - Now I have updated the retrieval service separately for explicit paper queries and general queries:
    - For explicit paper queries:
      - detect title -> filter article_id -> retrieve many chunks within that paper -> rerank chunks -> merge top chunks -> send one paper context to LLM
      - We still rerank chunks inside the paper to get the most relevant chunks (top 50 chunks) and then merge them to send to the LLM
    - For general queries:
      - retrieve broad chunk pool across corpus(, e.g. 100-200 chunks) -> rerank chunks -> group by article -> select top 5 papers(2-4 chunks per paper)
      - We retrieve a larger pool of chunks across the entire corpus (100-200 chunks for now) and then rerank them to get the most relevant chunks (top 50 for now). After reranking, we group the chunks by article and select the top (top 5 for now) papers based on the relevance of their chunks. This way, we ensure that we are not excluding potentially useful papers and are providing a more comprehensive context to the LLM for general queries.

---

## 2026-07-18

- Updated `indexing_service.py`:
  - Initially I was using the chunk index `i` for the `chunk_index` in the payload metadata when indexing the chunks into the vector database. However, this might cause issues since different sections of the paper can have same chunk index `i` (e.g. the first chunk of the abstract and the first chunk of the methods section will both have `i=0`). This can lead to confusion because in the retrieval service, I sort the retrieved chunks based on their `chunk_index` (before preparing the `merged_text`) to maintain the order of the chunks as they appear in the paper. If multiple chunks have the same `chunk_index`, it can lead to incorrect ordering of the chunks in the context provided to the LLM.
  - Now I have updated the `indexing_service.py` to use a `global_chunk_idx` which is incremented for each chunk across all sections of the paper. This ensures that each chunk has a unique `chunk_index` in the payload metadata, allowing for correct ordering of the chunks during retrieval and merging.
- Prepared the RAGAS evaluation powerpoint slides to present to the supervisor.

---

## 2026-07-24

- Updated `fetcher/article_fetcher.py`:
  - Initially I was not including any tabular data from the paper when fetching the contents. I thought they were in image format but now I realized they are actually in the html format and can be extracted as text. Now I have updated the `fetcher/article_fetcher.py` to extract the tabular data from the paper and include it right exactly in the order it appears in the paper. This was very necessary because many description only contains the main outcome from the tables and don't include the entire findings and this was causing the LLM to not have enough context to answer questions. Now with the tabular data included, the LLM has access to all the relevant information and can provide more accurate and comprehensive responses to queries.
  - Updated the article fetcher to also apply the latest table inclusion while fetching one or bulk of articles from the OpenAlex API
- Updated the frontend to align with the latest changes in the backend API and to include the new functionalities for fetching articles from OpenAlex API and indexing them into the vector database.

---

## 2026-08-01
- Updated `system_prompt.py` 
  - My previous system prompt was not clear and confused the LLM since I was mixing the instructions for explicit paper queries and general queries
  - In the updated system prompt, I have clearly separated all paper prompt, relevant paper prompt and general prompt.
- Updated `qdrant_client.py` to include a new method `get_all_article_ids()` which retrieves all the unique article ids from the vector database for general queries where we want to retrieve relevant chunks from all papers in the corpus.
- Updated the `retrieval_schema.py`
  - Defined `RetrievalScope` enum to specify the scope of the retrieval (e.g., ALL, SPECIFIC_PAPER, etc.)
  - Added `retrieval_scope` and `article_id` fields to the `QueryRequest` to be able to specify the scope of the retrieval 
- Updated `chat_schema.py` 
  - Added `retrieval_scope` to the `ChatRequest` and `ChatResponse` to allow users to specify the scope of the retrieval when asking questions.
- Updated `retrieval_service.py` 
  - Initially the retrieval service was only able to handle queries that either mentioned a specific paper title or general queries. But the general queries were not able to retrieve relevant chunks from each paper and were only able to retrieve chunks from a few papers. It was only returning a limited number of chunks (top 50) from the entire corpus and then grouping them by article to select the top 5 papers. This approach was not ideal for general queries as it could exclude relevant chunks from other papers. 
  - Now in the updated retrieval service:
    -  `_retrieve_across_all_papers` method first retrieves all article ids from the vector db.
      -  Loop through each article id and does the hybrid search for each article id to retrieve relevant chunks from each paper.
        - `_prepare_chunk_candidates()` takes the search results from each paper and a dictionary of payload, score
        -  each chunk candidate is then re-ranked
        -  then for each paper's re-ranked chunks, only the top 50 chunks is added to the `candidates_by_article` dictionary that store the re-ranked chunks for each paper.
      - After that `_build_response_from_grouped_chunks()` method is called to merge the top chunks from each paper and prepare the final response for the LLM.
        - For each article id and chunks in the candidates by article dictionary, the `llm_docs` is prepared by taking only `max_chunks_per_article` (top 2 for now) and merging them to prepare section-wise merged text.
        - It returns a `RetrievedDocument` object for each paper with the merged text and metadata.
- Updated the `pipeline.py` 
  - The `ChatResponse` and the `QueryRequest` now includes the `retrieval_scope` field to indicate whether the response was generated from a specific paper or across all papers in the corpus.
  - To improve the UI experience, the `_markdown_table_cell()` creates a markdown table cell for each source to create a table of source documents in the response. This allows the user to easily see which papers were used as sources for the response and provides a clear reference for further reading or verification of the information provided by the LLM.



## Meeting with supervisor

- Until which layer does the search happens in case of dense search and sparse search? Is it happening in the embedding layer or the vector database layer?
- Clearly describe what we are able to achieve in the project.
  - Specific list of questions to each individual paper : the output should be a table with the question, answer, and source document
    - Ex: does the paper address ICU or Non-ICU patients? Ans: For each paper, the answer should be whether it addresses ICU or Non-ICU patients and the source document should be the paper name.
    - Improve validation
      - Extract ground truth from at least 10 papers
      - Formulate a list of questions to ask to all the papers and get the answers in a structured format
      - First check what questions performed poorly and try to understand : either the question isn't formatted well or the ground truth is not provided properly.  
        - Either remove the questions or reformat the questions and re-evaluate the RAG pipeline 
        - Adjust ground truth or format the questions 
      
  - Should be able to query across all the papers (or selected papers) and get the answer with the source document
    - Ex: What papers address neutropenic fever? Ans: These three paper addresses neutropenic fever and the source documents should be the paper names.
    

- domain specific llm model vs general llm model
- [x] include tables from the paper as well


- PREVIOUS WORK: 
  - WHY IS IT NOT SUITED FOR MY QUESTION AND WHY MY METHOD MORE SUITED?
  - ARE THERE OTHER EXISTING TOOLS OTHER THAN WHAT WE ARE DOING? IF YES, BENCHMARK AND TALK ABOUT THEM.
  - REFERENCES AND TITLES
- Document everything so far.
- References for the methodological choices
- Scores (for all the queries present in the sheet) from the evaluation of the RAG pipeline
- See if other online LLM can extract those information from the paper: "Development and validation of machine learning models for predicting extubation failure in patients undergoing cardiac surgery: a retrospective study" 
- Why we chose openalex for fetching papers based on keywords:
- Duble check what happens to the tables from the extracted contents.
- How are the points being stored as graph in the vector database? Are they being stored as individual chunks with their own metadata or are they being grouped together based on the article they belong to?
- Look if we could build a knowledge base (like a vocabulary for searching) that would help give additional contexts for the LLMs. 


- A way to ask a list of questions to all the papers and get the answers in a structured format
- Prompt -> Answers from each paper
- Scrap papers based on the keywords (a list of papers) -> 10 papers  
- Prompt the queries( from the google sheet) -> Evaluate the answers
- Add the info about the mimic-iv dataset in the context 
- Add the info about ICD taxonomy in the context
- Reproducibility of papers -> [target (do they include ), cohort definition(do they include the ICD codes, OPS code)]
- Implement late chunking and for this use a bigger context sized window model like `bge-m3` (8k), `qwen3-embedding` (32k to 40k)