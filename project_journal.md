# Necessary commands
- `nix-shell` - activate environment with required libraries

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

- 

