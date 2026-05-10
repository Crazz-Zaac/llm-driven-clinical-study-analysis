import logging

from app.schemas.scrape_schema import (
    ScrapTextRequest,
    ScrapTextResponse,
    ScrapTextBatchRequest,
    ScrapTextBatchResponse,
    ArticleSchema,
    SectionSchema,
    MetadataSchema,
)
from app.scrapper.article_scrapper import ArticleScraper
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)


class ScrapTextService:
    """Service for scraping text from documents."""

    def __init__(self):
        self.scraper = ArticleScraper()

    def scrap_text(
        self, request: ScrapTextRequest, save_to_disk: bool = False
    ) -> ScrapTextResponse:
        """
        Scrape text from a single URL.

        Args:
            request: ScrapTextRequest containing the URL to scrape
            save_to_disk: If True, save the scraped article to app/data/ with unique filename
        Returns:
            ScrapTextResponse with the scraped article data
        Raises:
            ValueError: If scraping fails or article is not accessible
        """
        try:
            url = str(request.url)
            logger.info(f"Scraping article from: {url}")

            # Check if article is accessible
            if not self.scraper.check_url_availability(url):
                raise ValueError(
                    f"Article URL is not accessible or returned error: {url}"
                )

            # Extract article ID from URL
            article_id = self.scraper.extract_article_id(url)

            # Get cleaned HTML
            html_content, article_title = self.scraper.extract_article_html(url)
            if not html_content:
                raise ValueError(f"Failed to extract HTML content from {url}")

            # Extract sections
            sections_dict = self.scraper.extract_sections(html_content)
            missing_sections = [
                key for key, value in sections_dict.items() if not value
            ]
            if missing_sections:
                raise ValueError(
                    f"Found missing [{', '.join(missing_sections)}] for article ID: {article_id}"
                )
            sections = SectionSchema(**sections_dict)

            # Create article response
            article = ArticleSchema(
                article_id=article_id,
                url=request.url,
                source="nature",
                scraped_at=datetime.now(),
                sections=sections,
            )

            # Save to disk if requested
            if save_to_disk:
                article_data = {
                    "article_id": article_id,
                    "url": str(request.url),
                    "title": article_title,
                    "source": "nature",
                    "scraped_at": datetime.now().isoformat(),
                    **sections_dict,
                }
                self.scraper.save_article(article_data)

            message = f"Article scraped successfully: {article_id}"

            logger.info(f"Successfully scraped article: {article_id}")
            return ScrapTextResponse(article=article, message=message)

        except Exception as e:
            logger.error(f"Error scraping {request.url}: {str(e)}")
            raise

    def scrap_text_batch(
        self, request: ScrapTextBatchRequest, save_to_disk: bool = False
    ) -> ScrapTextBatchResponse:
        """
        Scrape text from multiple URLs.

        Args:
            request: ScrapTextBatchRequest containing URLs to scrape
            save_to_disk: If True, save scraped articles to data/scrapped_articles/
        Returns:
            ScrapTextBatchResponse with the scraped article data
        """
        articles: list[ArticleSchema] = []
        for url in request.urls:
            try:
                response = self.scrap_text(
                    ScrapTextRequest(url=url),
                    save_to_disk=save_to_disk,
                )
                articles.append(response.article)
            except Exception as e:
                logger.warning(f"Skipping URL due to scrape error: {url} ({str(e)})")

        return ScrapTextBatchResponse(articles=articles)

    def list_scraped_articles(self) -> list[dict[str, str | None]]:
        """List all scraped articles from the data/scrapped_articles/ directory."""
        from pathlib import Path
        import json

        article_dir = Path(settings.SCRAPED_ARTICLES_DIR)
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
