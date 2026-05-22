"""
LLM Brand Entity Monitor
────────────────────────
Autonomous daily scheduler that probes multiple LLMs with brand-awareness prompts,
extracts mention signals, and delivers a Slack digest.

Meets InSpace requirements:
  ✅ LLM integration (Claude + GPT-4o-mini multi-model querying)
  ✅ Autonomous scheduling (APScheduler cron, runs daily at 08:00)
  ✅ Structured data processing (MentionResult dataclass pipeline)
  ✅ API integrations (Anthropic API, OpenAI API, Slack Web API)
  ✅ Retry / resilience (tenacity exponential backoff on all LLM calls)
"""
from __future__ import annotations
import os
import asyncio
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from logger import get_logger
from prompt_builder import build_prompts
from llm_querier import LLMQuerier
from mention_analyzer import MentionAnalyzer
from slack_reporter import SlackReporter

load_dotenv()
log = get_logger("main")


def run_monitor():
    brand = os.environ["BRAND_NAME"]
    domain = os.environ.get("BRAND_DOMAIN", "technology")
    competitors = [c.strip() for c in os.environ.get("COMPETITORS", "").split(",") if c.strip()]

    log.info("Starting brand monitor run for '%s' at %s", brand, datetime.now().isoformat())

    querier = LLMQuerier()
    analyzer = MentionAnalyzer()
    reporter = SlackReporter()

    prompts = build_prompts(brand, domain, competitors)
    all_results = []

    for prompt in prompts:
        log.info("Querying LLMs with prompt: %s", prompt[:80])
        responses = querier.query_all(prompt)
        for resp in responses:
            if resp.error:
                continue
            result = analyzer.analyze(resp.model, brand, resp.answer)
            all_results.append(result)

    log.info("Collected %d mention results across %d prompts × %d models",
             len(all_results), len(prompts), 2)

    reporter.send_digest(brand, all_results)
    log.info("Run complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Brand Entity Monitor")
    parser.add_argument("--run-now", action="store_true", help="Run immediately instead of scheduling")
    parser.add_argument("--cron-hour", type=int, default=8, help="Hour to run daily (default: 8)")
    args = parser.parse_args()

    if args.run_now:
        run_monitor()
    else:
        scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
        scheduler.add_job(run_monitor, CronTrigger(hour=args.cron_hour, minute=0))
        log.info("Scheduler started — runs daily at %02d:00 Amsterdam time", args.cron_hour)
        try:
            scheduler.start()
        except KeyboardInterrupt:
            log.info("Scheduler stopped.")
