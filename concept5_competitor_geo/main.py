"""
Competitor GEO Intelligence Tracker
─────────────────────────────────────
Autonomous weekly scheduler that probes multiple LLMs with category-level
questions, extracts competitor brand visibility scores, stores history in
SQLite, and produces a Markdown competitive intelligence report.

Meets InSpace requirements:
  ✅ LLM integration (Claude Haiku + GPT-4o-mini multi-model probing)
  ✅ Autonomous scheduling (APScheduler weekly cron, Friday 06:00)
  ✅ Persistent data / trend tracking (SQLite history store)
  ✅ Structured reporting (Markdown with rankings + trend deltas)
  ✅ Retry / resilience (tenacity on all LLM calls)
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from logger import get_logger
from query_builder import build_queries
from geo_prober import GEOProber
from visibility_scorer import VisibilityScorer
from history_store import HistoryStore
from report_writer import ReportWriter

load_dotenv()
log = get_logger("main")


def run_tracker():
    my_brand = os.environ["MY_BRAND"]
    category = os.environ.get("CATEGORY", "AI automation")
    competitors_raw = os.environ.get("COMPETITORS", "")
    competitors = [c.strip() for c in competitors_raw.split(",") if c.strip()]
    all_brands = [my_brand] + [c for c in competitors if c != my_brand]
    use_cases_raw = os.environ.get("USE_CASES", "")
    use_cases = [u.strip() for u in use_cases_raw.split(",") if u.strip()]

    log.info("GEO Tracker starting — brand='%s' category='%s' competitors=%d",
             my_brand, category, len(competitors))

    queries = build_queries(category, use_cases)
    prober = GEOProber()
    scorer = VisibilityScorer()
    store = HistoryStore()
    writer = ReportWriter(store)

    probe_results = prober.probe_all(queries)
    report = scorer.score(category, all_brands, probe_results)
    store.save(report)

    markdown = writer.write(report, my_brand)

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"geo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filename.write_text(markdown, encoding="utf-8")
    log.info("Report written to %s", filename)
    print(markdown)

    store.close()
    log.info("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competitor GEO Intelligence Tracker")
    parser.add_argument("--run-now", action="store_true", help="Run immediately")
    args = parser.parse_args()

    if args.run_now:
        run_tracker()
    else:
        scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
        scheduler.add_job(run_tracker, CronTrigger(day_of_week="fri", hour=6, minute=0))
        log.info("Scheduler started — runs every Friday at 06:00 Amsterdam time")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            log.info("Scheduler stopped.")
