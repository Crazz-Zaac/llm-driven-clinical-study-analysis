import time
import random
import re
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import trafilatura

from app.core.config import settings

logger = logging.getLogger(__name__)


class ArticleFetcher:
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

    def extract_article_html(self, url: str):
        """
        Extract and clean article HTML.

        Returns a 3-tuple: (cleaned_html, title, raw_html).
        raw_html is the untouched page source, kept around so callers can go
        looking for things trafilatura strips out (e.g. hyperlinks, since we
        call it with include_links=False below). Full-size-table links are
        exactly this kind of thing on Nature/Springer article pages.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            metadata = trafilatura.extract_metadata(response.text)
            title = metadata.title if metadata and metadata.title else "No Title"

            cleaned_html = trafilatura.extract(
                response.text,
                output_format="html",
                include_links=False,
                favor_precision=True,
                include_tables=True,
                include_comments=False,
            )

            return cleaned_html, title, response.text
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}: {e}")
            return "", "", ""
        except Exception as e:
            print(f"Other error for {url}: {e}")
            return "", "", ""

    def extract_sections(self, html_content: str) -> dict:

        if not html_content:
            return {"abstract": "", "methods": "", "results": "", "conclusion": ""}

        soup = BeautifulSoup(html_content, "html.parser")

        section_patterns = {
            "abstract": ["abstract", "summary"],
            "methods": ["method", "methods", "materials and methods", "methodology"],
            "results": ["results", "findings"],
            "discussion": ["discussion", "discussions"],
            "conclusion": ["conclusion", "conclusions"],
        }
        sections = {key: [] for key in section_patterns}
        current_section = None

        for elem in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):

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

        if not sections["abstract"]:
            abstract_elem = soup.find("div", {"class": re.compile("abstract", re.I)})
            if abstract_elem:
                sections["abstract"].append(abstract_elem.get_text(strip=True))

        return {k: "\n\n".join(v) for k, v in sections.items()}

    def _find_article_root(self, soup: BeautifulSoup):
        """
        Narrow a full page soup down to just the article body, so the
        section walk below doesn't pick up nav menus, "related articles"
        rails, or footer links that happen to contain words like
        "summary" or "discussion".
        """
        return (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"article.?body", re.I))
            or soup.find("main")
            or soup
        )

    def extract_sections_with_tables(self, raw_html: str, base_url: str):
        """
        Walk the raw article HTML once, in document order, to build the
        section text AND locate any "Full size table" links at the same
        time — then fetch those tables and splice their content back in
        at the exact point they were referenced.

        Returns (sections: dict[str, str], external_tables: list[dict]).
        Each external_tables entry also gets a "section" key recording
        where it was spliced in, for anything downstream that wants the
        raw/structured table separate from the flattened text.
        """
        empty_sections = {
            "abstract": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
        }
        if not raw_html:
            return empty_sections, []

        soup = BeautifulSoup(raw_html, "html.parser")
        root = self._find_article_root(soup)

        section_patterns = {
            "abstract": ["abstract", "summary"],
            "methods": ["method", "methods", "materials and methods", "methodology"],
            "results": ["results", "findings"],
            "discussion": ["discussion", "discussions"],
            "conclusion": ["conclusion", "conclusions"],
        }
        sections = {key: [] for key in section_patterns}
        current_section = None
        pending_links = []  # ordered list of {"placeholder", "url", "label", "section"}
        seen_urls = set()

        for elem in root.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table", "a"]):

            if elem.name in ["h1", "h2"]:
                heading_text = elem.get_text(" ", strip=True).lower()
                matched = False
                for section, patterns in section_patterns.items():
                    if any(
                        re.search(rf"\b{re.escape(p)}\b", heading_text)
                        for p in patterns
                    ):
                        current_section = section
                        matched = True
                        break
                if not matched:
                    current_section = None
                continue

            if not current_section:
                continue

            if elem.name == "a":
                href = elem.get("href", "")
                if not href:
                    continue
                link_text = elem.get_text(" ", strip=True)
                is_table_url = bool(re.search(r"/tables/\d+", href))
                is_table_text = bool(re.search(r"full\s*size\s*table", link_text, re.I))
                if not (is_table_url or is_table_text):
                    continue

                full_url = urljoin(base_url, href)
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                # Look for a "Table N" label within the immediate container
                # (the caption/link usually live together in one card/figure)
                # rather than searching the whole document backwards

                container = elem.find_parent(["figure", "div"]) or elem
                label_match = re.search(
                    r"Table\s*\d+", container.get_text(" ", strip=True), re.I
                )
                if label_match:
                    label = label_match.group(0)
                else:
                    num_match = re.search(r"/tables/(\d+)", href)
                    label = f"Table {num_match.group(1)}" if num_match else "Table"

                placeholder = f"[[TABLE_PLACEHOLDER_{len(pending_links)}]]"
                sections[current_section].append(placeholder)
                pending_links.append(
                    {
                        "placeholder": placeholder,
                        "url": full_url,
                        "label": label,
                        "section": current_section,
                    }
                )
                continue

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

        if not sections["abstract"]:
            abstract_elem = soup.find("div", {"class": re.compile("abstract", re.I)})
            if abstract_elem:
                sections["abstract"].append(abstract_elem.get_text(strip=True))

        # Fetch each linked table in the order it was found, then splice its
        # content back into the exact placeholder slot we left for it.
        external_tables = []
        for link in pending_links:
            table_data = self.fetch_full_size_table(link["url"])
            if table_data:
                table_data["label"] = link["label"]
                table_data["section"] = link["section"]
                external_tables.append(table_data)

                block = table_data["label"]
                if table_data.get("caption"):
                    block += f": {table_data['caption']}"
                block += f"\n{table_data['text']}"
                if table_data.get("footnotes"):
                    block += f"\nNotes: {table_data['footnotes']}"
            else:
                # Couldn't fetch it — drop the placeholder rather than
                # leaving a dangling marker in the text.
                block = ""

            section_list = sections[link["section"]]
            for i, item in enumerate(section_list):
                if item == link["placeholder"]:
                    section_list[i] = block
                    break

            time.sleep(random.uniform(0.5, 1.5))

        joined_sections = {
            k: "\n\n".join(v for v in vals if v) for k, vals in sections.items()
        }
        return joined_sections, external_tables

    def extract_table_links(self, raw_html: str, base_url: str) -> list:
        """
        Legacy standalone helper — kept for anyone calling it directly
        Find "Full size table" links on a Nature/Springer article page.

        These point to standalone pages like:
            https://www.nature.com/articles/{id}/tables/{n}
        and are only present in the raw HTML — trafilatura's include_links=False
        strips them out, so we must look for them before that step.

        Returns a list of dicts: [{"label": "Table 1", "url": "..."}, ...]
        """
        if not raw_html:
            return []

        soup = BeautifulSoup(raw_html, "html.parser")
        table_links = []
        seen_urls = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            link_text = a.get_text(" ", strip=True)

            looks_like_table_url = bool(re.search(r"/tables/\d+", href))
            looks_like_table_text = bool(
                re.search(r"full\s*size\s*table", link_text, re.I)
            )

            if not (looks_like_table_url or looks_like_table_text):
                continue

            full_url = urljoin(base_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Try to recover a label like "Table 1" from context: nearby
            # heading, the link's own text, or the URL's trailing number.
            label = None
            if link_text and not looks_like_table_text:
                label = link_text
            else:
                heading = a.find_previous(["h2", "h3", "h4", "p"])
                if heading:
                    heading_text = heading.get_text(" ", strip=True)
                    match = re.search(r"Table\s*\d+", heading_text, re.I)
                    if match:
                        label = match.group(0)
            if not label:
                num_match = re.search(r"/tables/(\d+)", href)
                label = f"Table {num_match.group(1)}" if num_match else "Table"

            table_links.append({"label": label, "url": full_url})

        return table_links

    def fetch_full_size_table(self, table_url: str, timeout: int = 10) -> dict:
        """
        Fetch a standalone Nature "full size table" page and pull out its
        caption, HTML table, and any footnotes.

        Returns a dict: {"url", "caption", "html", "text", "footnotes"}
        or {} if nothing could be extracted.
        """
        try:
            response = self.session.get(table_url, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch table page {table_url}: {e}")
            return {}

        soup = BeautifulSoup(response.text, "html.parser")

        # Caption: Nature table pages usually have it in a <h1> or a
        # <p>/<figcaption> classed something like "c-article-table-caption".
        caption = ""
        caption_elem = (
            soup.find(class_=re.compile("table.*caption", re.I))
            or soup.find("figcaption")
            or soup.find("h1")
        )
        if caption_elem:
            caption = caption_elem.get_text(" ", strip=True)

        # The table itself.
        table_elem = soup.find("table")
        if not table_elem:
            logger.info(f"No <table> element found on {table_url}")
            return {}

        table_html = str(table_elem)

        rows = []
        for row in table_elem.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])
            ]
            if cells:
                rows.append(" | ".join(cells))
        table_text = "\n".join(rows)

        # Footnotes, e.g. significance markers, sample-size notes, etc.
        footnote_elems = soup.find_all(class_=re.compile("footnote|table.*foot", re.I))
        footnotes = "\n".join(
            fe.get_text(" ", strip=True) for fe in footnote_elems if fe.get_text(strip=True)
        )

        return {
            "url": table_url,
            "caption": caption,
            "html": table_html,
            "text": table_text,
            "footnotes": footnotes,
        }

    def fetch_external_tables(self, raw_html: str, base_url: str, delay_range=(0.5, 1.5)) -> list:
        """
        Find and fetch every full-size table linked from an article page.
        A small delay is added between requests to be polite to the server.
        """
        links = self.extract_table_links(raw_html, base_url)
        tables = []
        for link in links:
            table_data = self.fetch_full_size_table(link["url"])
            if table_data:
                table_data["label"] = link["label"]
                tables.append(table_data)
            time.sleep(random.uniform(*delay_range))
        return tables

    def extract_article_id(self, url: str) -> str:
        """Extract article ID from URL"""
        match = re.search(r"/articles/([^/?]+)", url)
        return match.group(1) if match else url.split("/")[-1]

    def save_article(self, output_data: dict):
        """Save article sections to a JSON file with unique name"""
        output_dir = Path(settings.FETCHED_ARTICLES_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)  # no more parents[1] hack

        article_id = output_data.get("article_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{article_id}_{timestamp}_{unique_id}.json"

        output_path = output_dir / filename
        logger.info(f"Saving to resolved path: {output_dir.resolve()}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Article saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save article to {output_path}: {str(e)}")
            raise

    def process_article(self, url: str):
        article_id = self.extract_article_id(url)
        html_content, article_title, raw_html = self.extract_article_html(url)

        if not raw_html:
            print(f"Failed to extract HTML from {url}")
            return

        # Single pass over the raw article HTML: builds section text and
        # fetches/splices in any "full size table" pages at the exact point
        # they're referenced, so a table ends up in the same section (and
        # same relative position) as it appears in the actual article.
        sections, external_tables = self.extract_sections_with_tables(raw_html, url)

        non_empty = {k: v for k, v in sections.items() if v}
        if non_empty or external_tables:
            logger.info(f"Saving contents for article ID: {article_id}")
            fetched_contents = {
                "article_id": article_id,
                "url": url,
                "title": article_title,
                **sections,  # empty sections will just be empty strings;
                # tables are already inlined here, in the right section's
                # text, at the point they were referenced -- no separate
                # "external_tables" list, since that was just a duplicate
                # copy of what's already embedded above.
            }
        else:
            logger.info(f"No content extracted for {article_id}, skipping.")
            return None
        return fetched_contents