import logging

from app.schemas.scrape_schema import (
    ScrapTextRequest,
    ScrapTextResponse,
    ArticleSchema,
    SectionSchema,
    MetadataSchema,
)
from app.scrapper.article_scrapper import ArticleScraper
from datetime import datetime


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
            html_content = self.scraper.extract_article_html(url)
            if not html_content:
                raise ValueError(f"Failed to extract HTML content from {url}")

            # Extract sections
            sections_dict = self.scraper.extract_sections(html_content)
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
