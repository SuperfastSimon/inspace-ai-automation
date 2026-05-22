from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import requests
import extruct
from bs4 import BeautifulSoup
from w3lib.html import get_base_url
from logger import get_logger

log = get_logger("page_analyzer")

@dataclass
class PageAnalysis:
    url: str
    title: str
    meta_description: str
    body_text: str
    existing_schema_types: list[str] = field(default_factory=list)
    page_type_hint: str = "Article"

    @property
    def needs_schema(self) -> bool:
        return len(self.existing_schema_types) == 0


class PageAnalyzer:
    TYPE_HINTS = {"product": "Product", "shop": "Product", "faq": "FAQPage",
                  "question": "FAQPage", "blog": "Article", "news": "Article"}

    def fetch(self, url: str) -> Optional[PageAnalysis]:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "SchemaBot/1.0"})
            resp.raise_for_status()
        except Exception as exc:
            log.error("Failed to fetch %s: %s", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.string.strip() if soup.title else ""
        meta = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta["content"].strip() if meta and meta.get("content") else ""
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        body_text = soup.get_text(" ", strip=True)[:3000]

        base = get_base_url(resp.text, resp.url)
        data = extruct.extract(resp.text, base_url=base, syntaxes=["json-ld"])
        existing = [item.get("@type", "") for item in data.get("json-ld", []) if "@type" in item]

        hint = self._infer_page_type(url, title)
        return PageAnalysis(url=url, title=title, meta_description=meta_desc,
                            body_text=body_text, existing_schema_types=existing,
                            page_type_hint=hint)

    def _infer_page_type(self, url: str, title: str) -> str:
        combined = (url + " " + title).lower()
        for keyword, ptype in self.TYPE_HINTS.items():
            if keyword in combined:
                return ptype
        return "Article"
