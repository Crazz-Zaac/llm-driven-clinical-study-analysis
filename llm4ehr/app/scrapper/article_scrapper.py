import time
import random
import re
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import trafilatura


logger = logging.getLogger(__name__)


class ArticleScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    def check_url_availability(self, url: str, timeout: int = 10) -> bool:
        """
        Check if URL is accessible and returns successful status code.

        Args:
            url: URL to check
            timeout: Request timeout in seconds
        Returns:
            True if URL is accessible (200-299 status), False otherwise
        """
        try:
            response = self.session.head(url, timeout=timeout, allow_redirects=True)
            is_available = 200 <= response.status_code < 300

            if is_available:
                logger.debug(
                    f"URL availability check passed: {url} (status: {response.status_code})"
                )
            else:
                logger.warning(
                    f"URL returned error status: {url} (status: {response.status_code})"
                )

            return is_available

        except requests.exceptions.Timeout:
            logger.error(f"URL availability check timed out: {url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"URL availability check failed: {url} - {str(e)}")
            return False

    def extract_article_html(self, url: str) -> str:
        """Extract and clean article HTML"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Use trafilatura to clean the HTML but keep structure
            cleaned_html = trafilatura.extract(
                response.text,
                output_format="html",
                include_links=False,
                favor_precision=True,
                include_tables=True,
                include_comments=False,
            )

            return cleaned_html
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}: {e}")
            return ""
        except Exception as e:
            print(f"Other error for {url}: {e}")
            return ""

    def extract_sections(self, html_content: str) -> dict:

        if not html_content:
            return {"abstract": "", "methods": "", "results": "", "conclusion": ""}

        soup = BeautifulSoup(html_content, "html.parser")

        section_patterns = {
            "abstract": ["abstract", "summary"],
            "methods": ["method", "methods", "materials and methods", "methodology"],
            "results": ["results", "findings"],
            "conclusion": ["conclusion", "discussion", "discussions", "conclusions"],
        }
        sections = {key: [] for key in section_patterns}
        current_section = None

        #  iterate through document elements in order
        for elem in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):

            # ---- Detect new main section ----
            if elem.name in ["h1", "h2"]:
                heading_text = elem.get_text(" ", strip=True).lower()
                for section, patterns in section_patterns.items():
                    if any(
                        re.search(rf"\b{re.escape(p)}\b", heading_text)
                        for p in patterns
                    ):
                        current_section = section
                        break
                else:
                    current_section = None

            # ---- Collect content if inside section ----
            elif current_section:

                if elem.name == "p":
                    text = elem.get_text(strip=True)
                    if text:
                        sections[current_section].append(text)

                elif elem.name in ["ul", "ol"]:
                    for li in elem.find_all("li"):
                        text = li.get_text(strip=True)
                        if text:
                            sections[current_section].append(f"• {text}")

                elif elem.name == "table":
                    rows = []
                    for row in elem.find_all("tr"):
                        cells = [
                            cell.get_text(strip=True)
                            for cell in row.find_all(["td", "th"])
                        ]
                        if cells:
                            rows.append(" | ".join(cells))
                    if rows:
                        sections[current_section].append("\n".join(rows))

        # fallback abstract detection
        if not sections["abstract"]:
            abstract_elem = soup.find("div", {"class": re.compile("abstract", re.I)})
            if abstract_elem:
                sections["abstract"].append(abstract_elem.get_text(strip=True))

        return {k: "\n\n".join(v) for k, v in sections.items()}

    def extract_article_id(self, url: str) -> str:
        """Extract article ID from URL"""
        match = re.search(r"/articles/([^/?]+)", url)
        return match.group(1) if match else url.split("/")[-1]

    def save_article(self, output_data: dict):
        """Save article sections to a JSON file with unique name"""
        # Create data directory inside app (same level as scrapper/)
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)

        # Generate unique filename: article_id_timestamp_uuid.json
        article_id = output_data.get("article_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{article_id}_{timestamp}_{unique_id}.json"

        output_path = output_dir / filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Article saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save article to {output_path}: {str(e)}")
            raise

    def process_article(self, url: str):

        # Extract article ID
        article_id = self.extract_article_id(url)

        # Get cleaned HTML
        html_content = self.extract_article_html(url)
        if not html_content:
            print(f"Failed to extract HTML from {url}")
            return

        # Extract sections
        sections = self.extract_sections(html_content)

        scrapped_contents = {}
        # Check if all the sections have content
        if all(sections.values()):
            print(f"Saving contents for article ID: {article_id}")
            scrapped_contents = {"article_id": article_id, "url": url, **sections}
        else:
            print(
                f"Found missing sections for article ID: {article_id}, skipping save."
            )
            return None
        return scrapped_contents


if __name__ == "__main__":
    scraper = ArticleScraper()
    test_url = "https://www.nature.com/articles/s41409-025-02761-5"
    article_data = scraper.process_article(test_url)
    if article_data:
        scraper.save_article(article_data)
