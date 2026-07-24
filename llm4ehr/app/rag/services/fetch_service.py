import logging
import json
from pathlib import Path

from app.schemas.fetch_schema import (
    FetchTextRequest,
    FetchTextResponse,
    FetchTextBatchRequest,
    FetchTextBatchResponse,
    FetchTextBatchFailure,
    ArticleSchema,
    SectionSchema,
    MetadataSchema,
)
from app.fetcher.article_fetcher import ArticleFetcher
from app.core.config import settings
from datetime import datetime

logger = logging.getLogger(__name__)


class FetchTextService:
    """Service for fetching text from documents."""

    def __init__(self):
        self.fetcher = ArticleFetcher()

    def fetch_text(
        self, request: FetchTextRequest, save_to_disk: bool = False
    ) -> FetchTextResponse:
        """
        Fetch text from a single URL.

        Args:
            request: FetchTextRequest containing the URL to fetch
            save_to_disk: If True, save the fetched article to app/data/ with unique filename
        Returns:
            FetchTextResponse with the fetched article data
        Raises:
            ValueError: If fetching fails or article is not accessible
        """
        try:
            url = str(request.url)
            logger.info(f"Fetching article from: {url}")

            if not self.fetcher.check_url_availability(url):
                raise ValueError(
                    f"Article URL is not accessible or returned error: {url}"
                )

            article_id = self.fetcher.extract_article_id(url)

            _html_content, article_title, raw_html = self.fetcher.extract_article_html(
                url
            )
            if not raw_html:
                raise ValueError(f"Failed to extract article HTML from {url}")

            # Keep this path aligned with ArticleFetcher.process_article so
            # direct URL fetches resolve Nature's linked full-size tables.
            sections_dict, _external_tables = self.fetcher.extract_sections_with_tables(
                raw_html, url
            )
            if not any(sections_dict.values()):
                raise ValueError(
                    f"No article sections could be extracted for article ID: {article_id}"
                )
            sections = SectionSchema(**sections_dict)

            article = ArticleSchema(
                article_id=article_id,
                url=request.url,
                source="nature",
                fetched_at=datetime.now(),
                sections=sections,
            )

            if save_to_disk:
                article_data = {
                    "article_id": article_id,
                    "url": str(request.url),
                    "title": article_title,
                    "source": "nature",
                    "fetched_at": datetime.now().isoformat(),
                    **sections_dict,
                }
                self.fetcher.save_article(article_data)

            message = f"Article fetched successfully: {article_id}"

            logger.info(f"Successfully fetched article: {article_id}")
            return FetchTextResponse(article=article, message=message)

        except Exception as e:
            logger.error(f"Error fetching {request.url}: {str(e)}")
            raise

    def fetch_text_batch(
        self, request: FetchTextBatchRequest, save_to_disk: bool = False
    ) -> FetchTextBatchResponse:
        """
        Fetch text from multiple URLs.

        Args:
            request: FetchTextBatchRequest containing URLs to fetch
            save_to_disk: If True, save fetched articles to data/fetched_articles/
        Returns:
            FetchTextBatchResponse with the fetched article data
        """
        articles: list[ArticleSchema] = []
        failed_urls: list[FetchTextBatchFailure] = []
        for url in request.urls:
            try:
                response = self.fetch_text(
                    FetchTextRequest(url=url),
                    save_to_disk=save_to_disk,
                )
                articles.append(response.article)
            except Exception as e:
                logger.warning(f"Skipping URL due to fetch error: {url} ({str(e)})")
                failed_urls.append(FetchTextBatchFailure(url=url, error=str(e)))

        return FetchTextBatchResponse(articles=articles, failed_urls=failed_urls)

    def list_fetched_articles(self) -> list[dict[str, str | None]]:
        """List all fetched articles from the data/fetched_articles/ directory."""
        article_dir = self._fetched_articles_dir()
        if not article_dir.exists():
            return []

        articles: list[dict[str, str | None]] = []
        seen: set[str] = set()
        for file in article_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    article_id = data.get("article_id", "")
                    if not article_id or article_id in seen:
                        continue
                    seen.add(article_id)
                    articles.append(
                        {
                            "article_id": article_id,
                            "title": data.get("title"),
                            "url": data.get("url", ""),
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to load article from {file}: {str(e)}")

        return articles

    def delete_fetched_articles(
        self, article_ids: list[str], delete_all: bool = False
    ) -> tuple[int, list[str]]:
        """Delete all stored JSON snapshots for selected fetched articles or all articles."""
        article_dir = self._fetched_articles_dir()
        if not article_dir.exists() or (not delete_all and not article_ids):
            return 0, []

        selected_ids = set(article_ids)
        deleted_article_ids: set[str] = set()
        deleted_files: list[str] = []
        unparseable_deleted_count = 0

        for file in article_dir.glob("*.json"):
            try:
                article_id = None
                if delete_all:
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            article_id = json.load(f).get("article_id")
                    except (OSError, ValueError) as e:
                        logger.warning("Deleting unreadable fetched article file %s: %s", file, e)
                else:
                    with open(file, "r", encoding="utf-8") as f:
                        article_id = json.load(f).get("article_id")
                        if article_id not in selected_ids:
                            continue
                file.unlink()
                deleted_files.append(str(file))
                if article_id:
                    deleted_article_ids.add(article_id)
                else:
                    unparseable_deleted_count += 1
            except Exception as e:
                logger.error("Failed to delete fetched article file %s: %s", file, e)

        return len(deleted_article_ids) + unparseable_deleted_count, deleted_files

    @staticmethod
    def _fetched_articles_dir() -> Path:
        article_dir = Path(settings.FETCHED_ARTICLES_DIR)
        if not article_dir.is_absolute():
            return Path(__file__).resolve().parents[3] / article_dir
        return article_dir
