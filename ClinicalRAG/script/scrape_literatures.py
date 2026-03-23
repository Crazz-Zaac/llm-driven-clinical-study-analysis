import time
import random
import re
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import trafilatura


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

    def scrape_articles(self, query: str, max_pages: int = 5) -> list:
        search_url = "https://www.nature.com/search"
        article_links = []

        for page in range(1, max_pages + 1):
            params = {"q": query, "page": page}
            response = self.session.get(search_url, params=params)
            if response.status_code != 200:
                print(f"Failed to retrieve search results for page {page}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Nature.com specific selectors
            for link in soup.select('a[data-track-action="view article"]'):
                href = link.get("href")
                if href and "/articles/" in href:
                    full_url = (
                        "https://www.nature.com" + href
                        if href.startswith("/")
                        else href
                    )
                    article_links.append(full_url)

            print(f"Page {page}: Found {len(article_links)} articles so far")

        return article_links

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
        """Append article sections to a JSONL file"""
        output_dir = Path(__file__).parent.parent / "dataset"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "scraped_articles.jsonl"

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

        """Process a single article end-to-end"""
        print(f"\n📄 Processing: {url}")
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
    query = "MIMIC-IV AND Retrospective"
    print(f"🔍 Searching for: '{query}'")
    article_urls = scraper.scrape_articles(query, max_pages=1)
    print(f"\n📊 Found {len(article_urls)} articles")

    for i, url in enumerate(article_urls, 1):
        print(f"\n--- Article {i}/{len(article_urls)} ---")
        content = scraper.process_article(url)
        if content:
            scraper.save_article(content)

        # Polite delay between requests
        if i < len(article_urls):
            delay = random.uniform(2, 5)
            print(f"⏱️  Waiting {delay:.1f} seconds...")
            time.sleep(delay)

        # if i == 5:
        #     break  # Limit to first 5 articles for testing
