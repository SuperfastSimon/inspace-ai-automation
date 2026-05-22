"""Fetch post list and metadata from WordPress REST API."""
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import get_logger

log = get_logger("cms_fetcher")


@dataclass
class PostMeta:
    post_id: int
    title: str
    url: str
    modified: datetime
    published: datetime
    categories: list[str]
    word_count: int = 0


class CMSFetcher:
    def __init__(self):
        self._base = os.environ["WP_SITE_URL"].rstrip("/")
        self._auth = (os.environ["WP_USERNAME"], os.environ["WP_APP_PASSWORD"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def fetch_all_posts(self, per_page: int = 100) -> list[PostMeta]:
        posts: list[PostMeta] = []
        page = 1
        while True:
            resp = requests.get(
                f"{self._base}/wp-json/wp/v2/posts",
                params={"per_page": per_page, "page": page, "status": "publish"},
                auth=self._auth, timeout=30,
            )
            if resp.status_code == 400:
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for p in batch:
                cats = [str(c) for c in p.get("categories", [])]
                posts.append(PostMeta(
                    post_id=p["id"],
                    title=p["title"]["rendered"],
                    url=p["link"],
                    modified=datetime.fromisoformat(p["modified"].replace("Z", "+00:00")),
                    published=datetime.fromisoformat(p["date"].replace("Z", "+00:00")),
                    categories=cats,
                ))
            total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages:
                break
            page += 1
        log.info("Fetched %d posts from WordPress", len(posts))
        return posts

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def fetch_word_count(self, post: PostMeta) -> int:
        resp = requests.get(post.url, timeout=20)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return len(text.split())
