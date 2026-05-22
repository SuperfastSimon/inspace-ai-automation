"""
Content Freshness Guardian
──────────────────────────
Autonomous weekly scheduler that scans a WordPress CMS for stale content,
scores staleness with configurable rules, generates AI-powered update
recommendations via Claude, and emails a prioritised HTML digest.

Meets InSpace requirements:
  ✅ LLM integration (Claude Haiku for per-post refresh recommendations)
  ✅ Autonomous scheduling (APScheduler weekly cron, Monday 07:00)
  ✅ CMS API integration (WordPress REST API with Application Password auth)
  ✅ Structured data pipeline (PostMeta → StalenessScore → ReportRow)
  ✅ HTML email reporting (Jinja2 template, SMTP delivery)
"""
from __future__ import annotations
import os
import argparse
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from logger import get_logger
from cms_fetcher import CMSFetcher
from staleness_scorer import StalenessScorer
from ai_summarizer import AISummarizer
from email_reporter import EmailReporter

load_dotenv()
log = get_logger("main")

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
MIN_PRIORITY = os.environ.get("MIN_PRIORITY", "medium")


def run_check():
    log.info("Content Freshness Guardian starting run…")

    fetcher = CMSFetcher()
    scorer = StalenessScorer()
    summarizer = AISummarizer()
    reporter = EmailReporter()

    posts = fetcher.fetch_all_posts()
    log.info("Scoring %d posts…", len(posts))

    scored = []
    for post in posts:
        try:
            post.word_count = fetcher.fetch_word_count(post)
        except Exception as exc:
            log.warning("Could not fetch word count for '%s': %s", post.title, exc)

        result = scorer.score(post)
        if PRIORITY_ORDER.get(result.priority, 99) <= PRIORITY_ORDER.get(MIN_PRIORITY, 2):
            scored.append(result)

    log.info("%d posts meet minimum priority '%s'", len(scored), MIN_PRIORITY)

    if not scored:
        log.info("Nothing stale to report — skipping email.")
        return

    # Sort: critical first, then by days descending
    scored.sort(key=lambda s: (PRIORITY_ORDER[s.priority], -s.days_since_modified))

    with_recs = []
    for item in scored:
        try:
            rec = summarizer.recommend(item)
        except Exception as exc:
            log.warning("AI recommendation failed for '%s': %s", item.post.title, exc)
            rec = "Review and update this post manually."
        with_recs.append((item, rec))

    reporter.send(with_recs)
    log.info("Run complete — %d items reported.", len(with_recs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Freshness Guardian")
    parser.add_argument("--run-now", action="store_true", help="Run immediately")
    parser.add_argument("--min-priority", choices=["critical", "high", "medium", "low"],
                        default=os.environ.get("MIN_PRIORITY", "medium"))
    args = parser.parse_args()

    if args.min_priority:
        os.environ["MIN_PRIORITY"] = args.min_priority

    if args.run_now:
        run_check()
    else:
        scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
        scheduler.add_job(run_check, CronTrigger(day_of_week="mon", hour=7, minute=0))
        log.info("Scheduler started — runs every Monday at 07:00 Amsterdam time")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            log.info("Scheduler stopped.")
