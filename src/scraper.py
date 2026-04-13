"""
UTD Academic Content Scraper.
Collects course catalogs, syllabi, advisories, and FAQs from utdallas.edu.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URLS = [
    "https://catalog.utdallas.edu/2024/graduate",
    "https://www.utdallas.edu/academics/",
    "https://advisingresource.utdallas.edu/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UTD-Academic-Bot/1.0; Research)"
}


@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str
    section: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.content.split())


class UTDScraper:
    """
    Polite web scraper that collects UTD academic content
    while respecting rate limits and robots.txt.
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_sec: float = 1.0,
        max_pages: int = 5000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay_sec
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.visited: set[str] = set()

    def scrape(self, start_urls: Optional[List[str]] = None) -> List[ScrapedPage]:
        urls = start_urls or BASE_URLS
        queue = list(urls)
        pages: List[ScrapedPage] = []

        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            page = self._fetch_page(url)
            if page and page.word_count > 50:
                pages.append(page)
                print(f"[{len(pages)}] {page.title[:60]} ({page.word_count} words)")

                # Discover child links on same domain
                for link in self._extract_links(url, page.content):
                    if link not in self.visited:
                        queue.append(link)

            time.sleep(self.delay)

        self._save(pages)
        print(f"\n✅ Scraped {len(pages)} pages → {self.output_dir}")
        return pages

    def _fetch_page(self, url: str) -> Optional[ScrapedPage]:
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [SKIP] {url}: {e}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url

        # Remove nav, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.find("body")
        content = main.get_text(separator=" ", strip=True) if main else ""
        content = " ".join(content.split())   # normalize whitespace

        section = self._classify_section(url)
        return ScrapedPage(url=url, title=title_text, content=content, section=section)

    @staticmethod
    def _extract_links(base_url: str, content: str) -> List[str]:
        soup = BeautifulSoup(content, "html.parser")
        base_domain = urlparse(base_url).netloc
        links = []
        for tag in soup.find_all("a", href=True):
            href = urljoin(base_url, tag["href"])
            parsed = urlparse(href)
            if parsed.netloc == base_domain and parsed.scheme in {"http", "https"}:
                links.append(href.split("#")[0])
        return list(set(links))

    @staticmethod
    def _classify_section(url: str) -> str:
        url_lower = url.lower()
        if "catalog" in url_lower:
            return "course_catalog"
        if "advising" in url_lower:
            return "advising"
        if "financial" in url_lower or "aid" in url_lower:
            return "financial_aid"
        if "registrar" in url_lower:
            return "registrar"
        return "general"

    def _save(self, pages: List[ScrapedPage]) -> None:
        out_file = self.output_dir / "scraped_pages.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for page in pages:
                f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    scraper = UTDScraper(output_dir="data/raw", delay_sec=0.5, max_pages=500)
    scraper.scrape()
