import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from pathlib import Path
from lxml import etree
from pyeuropepmc import SearchClient
from loguru import logger
from tqdm import tqdm

from preprocess import preprocess

# PDF and OCR
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract



load_dotenv()  # Load environment variables from .env file

# ----------------------------
# Configuration
# ----------------------------
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")  # Replace with your email for Unpaywall API
OUTPUT_DIR = Path("papers")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xml,text/html,*/*",
}


# ----------------------------
# Helper Functions
# ----------------------------
def check_oa_status(doi: str):
    """Check Open Access status via Unpaywall API."""
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": UNPAYWALL_EMAIL}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {
        "is_oa": data.get("is_oa"),
        "oa_location": data.get("best_oa_location"),
    }


def download_file(url: str, path: Path):
    """Download PDF or XML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def safe_article_path(article_id: str, ext: str = "json") -> Path:
    """
    Convert DOI/PMC ID into a safe filename in OUTPUT_DIR
    Replaces slashes and other unsafe characters with underscores.
    """
    import re

    safe_name = re.sub(r"[^\w\-\.]", "_", article_id)
    return OUTPUT_DIR / f"{safe_name}.{ext}"


def parse_pmc_xml(xml_path: Path):
    """Parse PMC XML into structured sections."""
    with open(xml_path, "rb") as f:
        root = etree.parse(f).getroot()

    sections = {}

    # Abstract
    abstract = root.xpath("//abstract//text()")
    sections["abstract"] = " ".join(abstract).strip()

    # Body sections (Introduction, Methods, Results, Discussion)
    for sec in root.xpath("//sec"):
        title = sec.xpath("./title/text()")
        text = sec.xpath(".//p//text()")
        if title:
            sections[title[0].lower()] = " ".join(text).strip()

    return sections


def extract_pdf_text(pdf_path: Path):
    """Extract text from PDF, fallback to OCR if necessary."""
    try:
        text = extract_text(str(pdf_path))
        if len(text.strip()) < 100:  # likely scanned
            raise ValueError("PDF seems scanned, using OCR")
        return text
    except Exception:
        text = ""
        images = convert_from_path(str(pdf_path))
        for img in images:
            text += pytesseract.image_to_string(img)
        return text


# ----------------------------
# Main Workflow
# ----------------------------
def process_article(article):
    article_id = article.get("pmcid") or article.get("doi")
    doi = article.get("doi")
    title = article.get("title")
    abstract = article.get("abstractText", "")
    journal = article.get("journalTitle")
    pub_year = article.get("pubYear")

    full_text = {}

    # Skip if no DOI for OA check
    if doi:
        try:
            oa_info = check_oa_status(doi)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Unpaywall lookup failed for {doi}: {e}")
            oa_info = {"is_oa": False, "oa_location": None}

        if oa_info["is_oa"] and oa_info["oa_location"]:
            oa_loc = oa_info["oa_location"]
            try:
                
                # Prefer XML
                if oa_loc.get("url_for_xml"):
                    xml_path = safe_article_path(article_id, ext="xml")
                    download_file(oa_loc["url_for_xml"], xml_path)
                    full_text = parse_pmc_xml(xml_path)
                    full_text = {"full_text": text}
                
                # Fallback PDF
                elif oa_loc.get("url_for_pdf"):
                    pdf_path = safe_article_path(article_id, ext="pdf")
                    download_file(oa_loc["url_for_pdf"], pdf_path)
                    text = extract_pdf_text(pdf_path)
                    full_text = {"full_text": text}
                else:
                    logger.info(f"No downloadable OA content for {article_id}")
            except requests.exceptions.HTTPError as e:
                logger.warning(
                    f"Download failed for {article_id} ({e.response.status_code}): {e}"
                )
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download error for {article_id}: {e}")
        else:
            logger.info(f"Article {article_id} is not OA")
    else:
        logger.info(f"No DOI for article {article_id}")

    # Only save if we got downloadable content
    if not full_text:
        logger.info(f"Skipping {article_id} — no downloadable OA content")
        return None

    # ----- Text Preprocessing -----
    raw = full_text.get("full_text", "")
    preprocessed = preprocess(
        raw,
        remove_references=True,
        remove_captions=True,
        remove_citations=True,
        chunk=True,
        extract_entities=False,  # set True once en_core_sci_lg is installed
    )
    logger.info(
        f"{article_id}: {len(preprocessed['sections'])} sections, "
        f"{len(preprocessed['chunks'])} chunks"
    )

    # Store metadata + text
    output_json = {
        "article_id": article_id,
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "journal": journal,
        "publication_year": pub_year,
        "full_text": full_text,
        "preprocessed": {
            "cleaned_text": preprocessed["cleaned_text"],
            "sections": preprocessed["sections"],
            "chunks": preprocessed["chunks"],
        },
    }

    out_path = safe_article_path(article_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {article_id}")
    return output_json


# ----------------------------
# Europe PMC Search
# ----------------------------
def main():
    all_articles = []

    with SearchClient() as client:
        results = client.search(
            query="MIMIC-IV AND Retrospective",     # query to find relevant articles
            resultType="core",
            format="json",
            abstractText=True,  # Include abstracts in results
            pageSize=100,       # Fetch 100 results per page
        )

        articles = results["resultList"]["result"]
        for article in tqdm(articles, desc="Processing articles", unit="article"):
            processed = process_article(article)
            if processed is not None:
                all_articles.append(processed)

    # Save a lightweight master index (metadata only — no full text)
    index_entries = [
        {
            "article_id": a["article_id"],
            "title": a["title"],
            "abstract": a["abstract"],
            "doi": a["doi"],
            "journal": a["journal"],
            "publication_year": a["publication_year"],
            "num_sections": len(a["preprocessed"]["sections"]),
            "num_chunks": len(a["preprocessed"]["chunks"]),
        }
        for a in all_articles
    ]
    with open(OUTPUT_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved index with {len(index_entries)} articles")


if __name__ == "__main__":
    main()
