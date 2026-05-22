from __future__ import annotations
import os
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import get_logger

log = get_logger("content_embedder")

@dataclass
class PageContent:
    url: str
    title: str
    text: str
    embedding: list[float] | None = None


class ContentEmbedder:
    def __init__(self):
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def fetch_content(self, url: str) -> PageContent | None:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "LinkBot/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            title = soup.title.string.strip() if soup.title else url
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)[:4000]
            return PageContent(url=url, title=title, text=text)
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def embed_batch(self, pages: list[PageContent], batch_size: int = 50) -> list[PageContent]:
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            texts = [f"{p.title} {p.text[:500]}" for p in batch]
            resp = self._openai.embeddings.create(model="text-embedding-3-small", input=texts)
            for page, item in zip(batch, resp.data):
                page.embedding = item.embedding
            log.info("Embedded batch %d/%d", i + batch_size, len(pages))
        return pages
