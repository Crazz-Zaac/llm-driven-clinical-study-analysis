import uuid
import time
import random
import logging
import requests
from sqlalchemy.orm import Session

from app.fetcher import ArticleFetcher
from app.db import crud
from app.core.config import settings

logger = logging.getLogger(__name__)


class PaperFetcher:
    def __init__(self):
        self.fetcher = ArticleFetcher()

    def _build_openalex_filter(self, keywords: list[str]) -> str:
        keyword_query = " ".join(keywords)
        return (
            f"title_and_abstract.search:{keyword_query},"
            f"open_access.is_oa:true,"
            f"primary_location.source.publisher_lineage:p4310319965"
        )

    def _cursor_key(self, keywords: list[str]) -> str:
        return " AND ".join(sorted(keywords))

    def _resolve_doi_url(self, doi_url: str) -> str:
        """Follow DOI redirect to get actual article URL."""
        try:
            response = requests.head(
                doi_url,
                allow_redirects=True,
                timeout=10,
                headers=self.fetcher.session.headers,
            )
            return response.url
        except Exception as e:
            logger.warning(f"DOI resolution failed for {doi_url}: {e}")
            return doi_url

    def search_openalex(
        self, keywords: list[str], cursor: str = "*", per_page: int = 25
    ) -> dict:
        params = {
            "filter": self._build_openalex_filter(keywords),
            "per-page": per_page,
            "cursor": cursor,
            "mailto": settings.UNPAYWALL_EMAIL,
        }
        response = requests.get(
            "https://api.openalex.org/works", params=params, timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return {
            "papers": data.get("results", []),
            "next_cursor": data.get("meta", {}).get("next_cursor"),
            "total_available": data.get("meta", {}).get("count", 0),
        }

    def run(self, keywords: list[str], max_results: int, db: Session) -> dict:
        job_id = str(uuid.uuid4())[:8]
        query_key = self._cursor_key(keywords)
        crud.create_job(db, job_id, query_key)

        summary = {
            "total_available": 0,
            "total_found": 0,
            "already_fetched": 0,
            "unavailable": 0,
            "newly_fetched": 0,
        }
        fetched_papers = []
        skipped_papers = []

        try:
            cursor = crud.get_cursor(db, query_key)
            result = self.search_openalex(
                keywords, cursor, per_page=min(max_results, 25)
            )
            papers = result["papers"]
            next_cursor = result["next_cursor"]
            summary["total_available"] = result["total_available"]
            summary["total_found"] = len(papers)

            for paper in papers:
                doi = paper.get("doi", "").replace("https://doi.org/", "")
                title = paper.get("title", "Unknown")
                target_url = (
                    paper.get("primary_location", {}).get("landing_page_url") or ""
                )

                # 1. Skip already fetched
                if doi and crud.is_doi_fetched(db, doi):
                    summary["already_fetched"] += 1
                    skipped_papers.append(
                        {"doi": doi, "title": title, "reason": "already_fetched"}
                    )
                    continue

                # 2. Skip if no URL
                if not target_url:
                    summary["unavailable"] += 1
                    skipped_papers.append(
                        {"doi": doi, "title": title, "reason": "no_url"}
                    )
                    continue

                # 3. Check reachability
                resolved_url = self._resolve_doi_url(target_url)
                if not self.fetcher.check_url_availability(resolved_url):
                    summary["unavailable"] += 1
                    skipped_papers.append({"doi": doi, "title": title, "reason": "unavailable"})
                    continue

                # 4. Fetch and save
                data = self.fetcher.process_article(resolved_url)

                if data:
                    self.fetcher.save_article(data)
                    crud.mark_paper_fetched(db, doi=doi, title=title, url=resolved_url)
                    summary["newly_fetched"] += 1
                    fetched_papers.append(
                        {"doi": doi, "title": title, "url": resolved_url}
                    )
                else:
                    summary["unavailable"] += 1
                    skipped_papers.append(
                        {"doi": doi, "title": title, "reason": "extraction_failed"}
                    )

                time.sleep(random.uniform(1.5, 3.0))

            if next_cursor:
                crud.save_cursor(db, query_key, next_cursor)

            full_summary = {
                **summary,
                "job_id": job_id,
                "keywords": keywords,
                "cursor_used": cursor,
                "next_cursor": next_cursor,
                "fetched_papers": fetched_papers,
                "skipped_papers": skipped_papers,
            }
            crud.complete_job(db, job_id, full_summary)
            return full_summary

        except Exception as e:
            logger.error(f"Pipeline job {job_id} failed: {e}")
            crud.fail_job(db, job_id, str(e))
            raise
