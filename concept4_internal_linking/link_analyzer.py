from __future__ import annotations
from urllib.parse import urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
from logger import get_logger

log = get_logger("link_analyzer")


def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/")


class LinkAnalyzer:
    def __init__(self, known_urls: set[str]):
        self._known = {normalize(u) for u in known_urls}

    def build_graph(self, pages: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        outgoing: dict[str, set[str]] = {normalize(u): set() for u in pages}
        incoming: dict[str, set[str]] = {normalize(u): set() for u in pages}

        for url in pages:
            norm = normalize(url)
            try:
                resp = requests.get(url, timeout=15, headers={"User-Agent": "LinkBot/1.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                base = urlparse(url)
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/"):
                        href = f"{base.scheme}://{base.netloc}{href}"
                    href_norm = normalize(href)
                    if href_norm in self._known and href_norm != norm:
                        outgoing[norm].add(href_norm)
                        incoming.setdefault(href_norm, set()).add(norm)
            except Exception as exc:
                log.warning("Could not analyze links for %s: %s", url, exc)

        log.info("Built link graph: %d nodes", len(outgoing))
        return outgoing, incoming
