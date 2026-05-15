import logging

from app.schemas.fetch_schema import (
    FetchTextRequest,
    FetchTextResponse,
    FetchTextBatchRequest,
    FetchTextBatchResponse,
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

            html_content, article_title = self.fetcher.extract_article_html(url)
            if not html_content:
                raise ValueError(f"Failed to extract HTML content from {url}")

            sections_dict = self.fetcher.extract_sections(html_content)
            missing_sections = [
                key for key, value in sections_dict.items() if not value
            ]
            if missing_sections:
                raise ValueError(
                    f"Found missing [{', '.join(missing_sections)}] for article ID: {article_id}"
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
        for url in request.urls:
            try:
                response = self.fetch_text(
                    FetchTextRequest(url=url),
                    save_to_disk=save_to_disk,
                )
                articles.append(response.article)
            except Exception as e:
                logger.warning(f"Skipping URL due to fetch error: {url} ({str(e)})")

        return FetchTextBatchResponse(articles=articles)

    def list_fetched_articles(self) -> list[dict[str, str | None]]:
        """List all fetched articles from the data/fetched_articles/ directory."""
        from pathlib import Path
        import json

        article_dir = Path(settings.FETCHED_ARTICLES_DIR)
        if not article_dir.is_absolute():
            article_dir = Path(__file__).resolve().parents[3] / article_dir
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
