"""
Structured Data Auto-Injector
──────────────────────────────
Crawls a sitemap or URL list, uses Claude to generate Schema.org JSON-LD,
validates it, and injects it into WordPress pages idempotently.

Meets InSpace requirements:
  ✅ LLM integration (Claude Haiku for schema generation)
  ✅ CMS API integration (WordPress REST API + Application Passwords)
  ✅ Structured data pipeline (PageAnalysis → schema dict → ValidationResult)
  ✅ Idempotent automation (HTML comment markers prevent duplicates)
  ✅ Retry / resilience (tenacity on all network calls)
"""
from __future__ import annotations
import argparse
import time
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import requests
from logger import get_logger
from page_analyzer import PageAnalyzer
from schema_generator import SchemaGenerator
from schema_validator import SchemaValidator
from cms_injector import WordPressInjector

load_dotenv()
log = get_logger("main")


def parse_sitemap(url: str) -> list[str]:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]


def run(urls: list[str], dry_run: bool, sleep: float):
    analyzer = PageAnalyzer()
    generator = SchemaGenerator()
    validator = SchemaValidator()
    injector = WordPressInjector()

    stats = {"skipped": 0, "injected": 0, "failed": 0}

    for url in urls:
        log.info("Processing: %s", url)
        page = analyzer.fetch(url)
        if not page:
            stats["failed"] += 1
            continue
        if not page.needs_schema:
            log.info("Skipping (schema exists): %s", url)
            stats["skipped"] += 1
            continue
        try:
            schema = generator.generate(page)
            result = validator.validate(schema)
            if not result.valid:
                log.warning("Invalid schema for %s: %s", url, result.errors)
                stats["failed"] += 1
                continue
            if result.warnings:
                log.warning("Schema warnings for %s: %s", url, result.warnings)
            if dry_run:
                log.info("[DRY RUN] Would inject schema type=%s into %s",
                         schema.get("@type"), url)
                stats["injected"] += 1
            else:
                ok = injector.inject(url, schema)
                stats["injected" if ok else "failed"] += 1
        except Exception as exc:
            log.error("Error processing %s: %s", url, exc)
            stats["failed"] += 1
        time.sleep(sleep)

    log.info("Done. injected=%d skipped=%d failed=%d",
             stats["injected"], stats["skipped"], stats["failed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Data Auto-Injector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sitemap", help="Sitemap URL")
    group.add_argument("--urls", nargs="+", help="Space-separated URLs")
    group.add_argument("--urls-file", help="File with one URL per line")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if args.sitemap:
        urls = parse_sitemap(args.sitemap)
    elif args.urls:
        urls = args.urls
    else:
        with open(args.urls_file) as f:
            urls = [line.strip() for line in f if line.strip()]

    if args.limit:
        urls = urls[:args.limit]

    run(urls, dry_run=args.dry_run, sleep=args.sleep)
