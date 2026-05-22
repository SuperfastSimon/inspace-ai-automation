#!/usr/bin/env python3
"""
Genesis Pipeline — AI Research & Content Intelligence
Self-sufficient, single-file autonomous pipeline.

Stages:
  1. Web research (Jina Reader API)
  2. Three parallel LLM analyses (Executive Summary / Opportunities / Action Plan)
  3. Combine results
  4. Format as HTML email report
  5. Send via SMTP (or print to stdout in dry-run mode)

Usage:
  python genesis_pipeline.py --topic "Saikou.tech AI automation" --email you@example.com
  python genesis_pipeline.py --topic "n8n workflow trends" --email you@example.com --dry-run
  python genesis_pipeline.py --schedule  # runs every Friday at 06:00
"""

import argparse
import asyncio
import logging
import os
import smtplib
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import anthropic
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(name: str = "genesis") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler("genesis_pipeline.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log

logger = _setup_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
JINA_API_KEY      = os.getenv("JINA_API_KEY", "")          # optional — improves rate limit
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER         = os.getenv("SMTP_USER", "")
SMTP_PASS         = os.getenv("SMTP_PASS", "")
SMTP_FROM         = os.getenv("SMTP_FROM", SMTP_USER)

SONNET_MODEL      = "claude-sonnet-4-6"
OPUS_MODEL        = "claude-opus-4-7"
GPT_MODEL         = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    topic: str
    raw_content: str
    sources: list[str] = field(default_factory=list)

@dataclass
class AnalysisResult:
    executive_summary: str = ""
    opportunities: str = ""
    action_plan: str = ""
    error: Optional[str] = None

@dataclass
class PipelineRun:
    topic: str
    recipient_email: str
    research: Optional[ResearchResult] = None
    analysis: Optional[AnalysisResult] = None
    html_report: str = ""
    sent: bool = False
    started_at: datetime = field(default_factory=datetime.now)

# ---------------------------------------------------------------------------
# Stage 1 — Web Research (Jina Reader)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _jina_fetch(url: str, jina_key: str) -> str:
    headers = {"Accept": "text/plain"}
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"
    jina_url = f"https://r.jina.ai/{url}"
    resp = requests.get(jina_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text[:8000]  # cap at 8k chars per source

def _search_web(query: str, jina_key: str) -> tuple[str, list[str]]:
    """Use Jina Search to find relevant pages, then fetch top results."""
    search_url = f"https://s.jina.ai/{requests.utils.quote(query)}"
    headers = {"Accept": "application/json"}
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"
    try:
        resp = requests.get(search_url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", [])[:3]
        sources = [r.get("url", "") for r in results]
        combined = "\n\n".join(
            f"[Source: {r.get('url', 'unknown')}]\n{r.get('description', r.get('content', ''))[:2000]}"
            for r in results
        )
        return combined, sources
    except Exception as e:
        logger.warning(f"Jina search fallback — {e}")
        return f"Research topic: {query}\n(Web search unavailable — LLM will use training knowledge)", []

def run_research(topic: str) -> ResearchResult:
    logger.info(f"[Stage 1] Researching: {topic}")
    content, sources = _search_web(topic, JINA_API_KEY)
    return ResearchResult(topic=topic, raw_content=content, sources=sources)

# ---------------------------------------------------------------------------
# Stage 2 — Parallel LLM Analysis (3 passes)
# ---------------------------------------------------------------------------

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _claude(system: str, user: str, model: str = SONNET_MODEL, max_tokens: int = 1200) -> str:
    if not claude_client:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    msg = claude_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _gpt(system: str, user: str, model: str = GPT_MODEL, max_tokens: int = 800) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = openai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()

def _analyze_summary(topic: str, research: str) -> str:
    system = (
        "You are a senior market intelligence analyst. Produce a concise executive summary "
        "(3–5 paragraphs) of the topic based on the research provided. Focus on current state, "
        "key players, and market dynamics. Be factual and insightful."
    )
    user = f"Topic: {topic}\n\nResearch:\n{research}"
    return _claude(system, user, model=SONNET_MODEL)

def _analyze_opportunities(topic: str, research: str) -> str:
    system = (
        "You are a business strategist specialising in AI and automation. "
        "Identify exactly 5 concrete growth or competitive opportunities related to the topic. "
        "Format as a numbered list. Each item: bold title + 2-sentence explanation."
    )
    user = f"Topic: {topic}\n\nResearch:\n{research}"
    try:
        return _claude(system, user, model=SONNET_MODEL)
    except Exception:
        return _gpt(system, user)

def _analyze_action_plan(topic: str, research: str) -> str:
    system = (
        "You are a fractional CMO / GTM strategist. Create a 90-day action plan with 3 phases "
        "(Days 1–30, 31–60, 61–90). Each phase: 3 specific, measurable actions. "
        "Format clearly. Be actionable, not generic."
    )
    user = f"Topic: {topic}\n\nResearch:\n{research}"
    return _claude(system, user, model=SONNET_MODEL)

def run_analysis(research: ResearchResult) -> AnalysisResult:
    logger.info("[Stage 2] Running 3 parallel LLM analyses...")
    result = AnalysisResult()
    tasks = {
        "summary":      (_analyze_summary,      research.topic, research.raw_content),
        "opportunities":(_analyze_opportunities, research.topic, research.raw_content),
        "action_plan":  (_analyze_action_plan,   research.topic, research.raw_content),
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fn, *args): key
            for key, (fn, *args) in tasks.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                value = future.result()
                setattr(result, key if key != "summary" else "executive_summary", value)
                logger.info(f"  ✓ {key} complete")
            except Exception as e:
                logger.error(f"  ✗ {key} failed: {e}")
                result.error = str(e)
    return result

# ---------------------------------------------------------------------------
# Stage 3 — Combine
# ---------------------------------------------------------------------------

def combine_results(topic: str, analysis: AnalysisResult, sources: list[str]) -> str:
    separator = "\n\n" + "─" * 60 + "\n\n"
    parts = [
        f"## Executive Summary\n\n{analysis.executive_summary}",
        f"## Top 5 Opportunities\n\n{analysis.opportunities}",
        f"## 90-Day Action Plan\n\n{analysis.action_plan}",
    ]
    combined = separator.join(parts)
    if sources:
        src_block = "\n".join(f"- {s}" for s in sources if s)
        combined += f"\n\n## Sources\n\n{src_block}"
    return combined

# ---------------------------------------------------------------------------
# Stage 4 — Format HTML Report
# ---------------------------------------------------------------------------

_HTML_SYSTEM = textwrap.dedent("""
    You are an expert HTML email designer. Convert the provided Markdown research report
    into a polished, self-contained HTML email. Requirements:
    - Inline CSS only (no <style> blocks — email client compatibility)
    - Max width 680px, centered
    - Clean sans-serif font (Arial/Helvetica fallback)
    - Header banner with dark navy background (#0d1b2a) and white title
    - Section headings in deep blue (#1a3a5c)
    - Numbered opportunity items with light blue left border
    - Action plan phases in subtle grey cards
    - Footer with report timestamp and "Generated by Genesis Pipeline"
    - Return ONLY the complete HTML — no markdown, no explanation, no code fences
""").strip()

def format_html_report(topic: str, combined: str) -> str:
    logger.info("[Stage 4] Formatting HTML report with Opus...")
    user = f"Report Topic: {topic}\n\nReport Content:\n\n{combined}"
    try:
        html = _claude(_HTML_SYSTEM, user, model=OPUS_MODEL, max_tokens=4000)
        # Ensure it starts with <!DOCTYPE or <html
        if not html.strip().lower().startswith(("<!doctype", "<html")):
            html = f"<html><body><pre>{html}</pre></body></html>"
        return html
    except Exception as e:
        logger.error(f"HTML formatting failed: {e} — falling back to plain-text HTML")
        safe = combined.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<html><body><pre style='font-family:Arial;max-width:680px;margin:auto'>{safe}</pre></body></html>"

# ---------------------------------------------------------------------------
# Stage 5 — Send Email
# ---------------------------------------------------------------------------

def send_email(recipient: str, topic: str, html_body: str, dry_run: bool = False) -> bool:
    subject = f"Genesis Pipeline Report: {topic} — {datetime.now().strftime('%Y-%m-%d')}"
    if dry_run:
        logger.info(f"[Stage 5 DRY-RUN] Would send to {recipient}: {subject}")
        # Save locally instead
        out_path = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        out_path.write_text(html_body, encoding="utf-8")
        logger.info(f"  Report saved to {out_path}")
        return True

    if not SMTP_USER or not SMTP_PASS:
        logger.warning("[Stage 5] SMTP credentials not configured — saving report locally")
        out_path = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        out_path.write_text(html_body, encoding="utf-8")
        logger.info(f"  Report saved to {out_path}")
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_FROM or SMTP_USER
    msg["To"]      = recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM or SMTP_USER, [recipient], msg.as_string())
        logger.info(f"[Stage 5] Email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"[Stage 5] Email failed: {e}")
        return False

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(topic: str, recipient_email: str, dry_run: bool = False) -> PipelineRun:
    run = PipelineRun(topic=topic, recipient_email=recipient_email)
    logger.info(f"\n{'='*60}")
    logger.info(f"Genesis Pipeline starting — topic: {topic!r}")
    logger.info(f"{'='*60}\n")

    # Stage 1
    run.research = run_research(topic)
    logger.info(f"  Sources found: {len(run.research.sources)}")

    # Stage 2
    run.analysis = run_analysis(run.research)
    if run.analysis.error and not run.analysis.executive_summary:
        logger.error("Analysis failed completely — aborting")
        return run

    # Stage 3
    combined = combine_results(topic, run.analysis, run.research.sources)

    # Stage 4
    run.html_report = format_html_report(topic, combined)

    # Stage 5
    run.sent = send_email(recipient_email, topic, run.html_report, dry_run=dry_run)

    elapsed = (datetime.now() - run.started_at).total_seconds()
    logger.info(f"\nPipeline complete in {elapsed:.1f}s — sent={run.sent}\n")
    return run

# ---------------------------------------------------------------------------
# Scheduler (weekly Friday 06:00)
# ---------------------------------------------------------------------------

def run_scheduled(topic: str, recipient_email: str):
    scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="fri", hour=6, minute=0, timezone="Europe/Amsterdam"),
        args=[topic, recipient_email],
        id="genesis_weekly",
        replace_existing=True,
    )
    logger.info(f"Genesis Pipeline scheduled — every Friday 06:00 Amsterdam time")
    logger.info(f"  Topic: {topic!r}  |  Recipient: {recipient_email}")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genesis Pipeline — AI Research & Content Intelligence"
    )
    parser.add_argument("--topic",    required=True,  help="Brand or topic to research")
    parser.add_argument("--email",    required=True,  help="Recipient email address")
    parser.add_argument("--dry-run",  action="store_true", help="Skip email; save HTML locally")
    parser.add_argument("--schedule", action="store_true", help="Run on weekly Friday 06:00 cron")
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(args.topic, args.email)
    else:
        run_pipeline(args.topic, args.email, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
