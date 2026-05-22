"""
Internal Linking Intelligence Engine
──────────────────────────────────────
Crawls a sitemap, embeds all pages with OpenAI, finds semantically related
pages lacking internal links, generates AI anchor text via Claude, and
exports a prioritised opportunity report to Google Sheets (or CSV fallback).

Meets InSpace requirements:
  ✅ LLM integration (OpenAI embeddings + Claude anchor generation)
  ✅ Semantic similarity (vectorized cosine similarity via NumPy)
  ✅ CMS / web scraping (sitemap parsing, page content extraction)
  ✅ Google Sheets export (gspread with service account auth)
  ✅ Retry / resilience (tenacity on all API calls)
"""
from __future__ import annotations
import argparse
import os
from dotenv import load_dotenv
from logger import get_logger
from sitemap_parser import SitemapParser
from content_embedder import ContentEmbedder
from link_analyzer import LinkAnalyzer, normalize
from opportunity_finder import OpportunityFinder
from anchor_generator import AnchorGenerator
from exporter import SheetsExporter

load_dotenv()
log = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="Internal Linking Intelligence Engine")
    parser.add_argument("--sitemap", required=True, help="Sitemap URL")
    parser.add_argument("--limit", type=int, default=0, help="Max pages to process")
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--skip-anchors", action="store_true", help="Skip AI anchor generation")
    args = parser.parse_args()

    log.info("Starting Internal Linking Engine — sitemap: %s", args.sitemap)

    # Step 1: Parse sitemap
    parser_obj = SitemapParser()
    urls = parser_obj.parse(args.sitemap)
    if args.limit:
        urls = urls[:args.limit]
    log.info("Processing %d URLs", len(urls))

    # Step 2: Fetch + embed content
    embedder = ContentEmbedder()
    pages = [p for url in urls if (p := embedder.fetch_content(url)) is not None]
    pages = embedder.embed_batch(pages)

    # Step 3: Build link graph
    analyzer = LinkAnalyzer(known_urls={p.url for p in pages})
    outgoing, incoming = analyzer.build_graph([p.url for p in pages])

    # Step 4: Find opportunities
    finder = OpportunityFinder(min_similarity=args.min_similarity)
    opportunities = finder.find(pages, outgoing, incoming)
    log.info("Found %d opportunities", len(opportunities))

    # Step 5: Generate anchors
    if args.skip_anchors:
        with_anchors = [(o, f"learn more about {o.target.title[:30]}") for o in opportunities]
    else:
        anchor_gen = AnchorGenerator()
        with_anchors = []
        for opp in opportunities:
            try:
                anchor = anchor_gen.suggest(opp)
            except Exception as exc:
                log.warning("Anchor generation failed: %s", exc)
                anchor = opp.target.title[:40]
            with_anchors.append((opp, anchor))

    # Step 6: Export
    exporter = SheetsExporter()
    output = exporter.export(with_anchors)
    log.info("Export complete: %s", output)
    print(f"\nDone! Report: {output}")


if __name__ == "__main__":
    main()
