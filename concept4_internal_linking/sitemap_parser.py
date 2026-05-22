from __future__ import annotations
import xml.etree.ElementTree as ET
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import get_logger

log = get_logger("sitemap_parser")
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

class SitemapParser:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def parse(self, url: str) -> list[str]:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "LinkBot/1.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        if root.findall("sm:sitemap", NS):
            urls = []
            for sub in root.findall("sm:sitemap/sm:loc", NS):
                urls.extend(self.parse(sub.text))
            return urls
        locs = [loc.text for loc in root.findall(".//sm:loc", NS) if loc.text]
        log.info("Parsed %d URLs from %s", len(locs), url)
        return locs
