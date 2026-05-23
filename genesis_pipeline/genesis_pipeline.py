#!/usr/bin/env python3
"""
Genesis Pipeline — AI Research & Content Intelligence
Self-sufficient, single-file autonomous pipeline with multi-provider LLM support.

Providers: anthropic (default) | openai | google

Usage:
  python genesis_pipeline.py --topic "Saikou.tech" --email you@example.com
  python genesis_pipeline.py --topic "n8n trends" --email you@example.com --provider openai
  python genesis_pipeline.py --topic "AI SEO" --email you@example.com --provider google
  python genesis_pipeline.py --topic "..." --email you@example.com --dry-run
  python genesis_pipeline.py --topic "..." --email you@example.com --schedule
"""

import argparse
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

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
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
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
JINA_API_KEY      = os.getenv("JINA_API_KEY", "")
SMTP_HOST         = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER         = os.getenv("SMTP_USER", "")
SMTP_PASS         = os.getenv("SMTP_PASS", "")
SMTP_FROM         = os.getenv("SMTP_FROM", SMTP_USER)

# Model IDs per provider
PROVIDER_MODELS = {
    "anthropic": {
        "fast":   "claude-sonnet-4-6",
        "powerful": "claude-opus-4-7",
    },
    "openai": {
        "fast":   "gpt-4o-mini",
        "powerful": "gpt-4o",
    },
    "google": {
        "fast":   "gemini-2.0-flash",
        "powerful": "gemini-2.5-pro-preview-05-06",
    },
}

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
    provider: str = "anthropic"
    research: Optional[ResearchResult] = None
    analysis: Optional[AnalysisResult] = None
    html_report: str = ""
    sent: bool = False
    started_at: datetime = field(default_factory=datetime.now)

# ---------------------------------------------------------------------------
# Provider-agnostic LLM caller
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _call_llm(system: str, user: str, provider: str, tier: str = "fast", max_tokens: int = 1200) -> str:
    models = PROVIDER_MODELS[provider]
    model  = models[tier]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        return resp.choices[0].message.content.strip()

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        m = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
        )
        resp = m.generate_content(user)
        return resp.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------------------------------------------------------------------------
# Stage 1 — Web Research (Jina)
# ---------------------------------------------------------------------------

def _search_web(query: str) -> tuple[str, list[str]]:
    search_url = f"https://s.jina.ai/{requests.utils.quote(query)}"
    headers = {"Accept": "application/json"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        resp = requests.get(search_url, headers=headers, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("data", [])[:3]
        sources = [r.get("url", "") for r in results]
        combined = "\n\n".join(
            f"[Source: {r.get('url','unknown')}]\n{r.get('description', r.get('content',''))[:2000]}"
            for r in results
        )
        return combined, sources
    except Exception as e:
        logger.warning(f"Jina search unavailable — {e}")
        return f"Research topic: {query}\n(Web search unavailable — LLM will use training knowledge)", []

def run_research(topic: str) -> ResearchResult:
    logger.info(f"[Stage 1] Researching: {topic}")
    content, sources = _search_web(topic)
    return ResearchResult(topic=topic, raw_content=content, sources=sources)

# ---------------------------------------------------------------------------
# Stage 2 — Parallel LLM Analysis
# ---------------------------------------------------------------------------

def _analyze_summary(topic: str, research: str, provider: str) -> str:
    return _call_llm(
        "You are a senior market intelligence analyst. Write a concise executive summary "
        "(3–5 paragraphs) covering current state, key players, and market dynamics. Be factual.",
        f"Topic: {topic}\n\nResearch:\n{research}",
        provider, tier="fast",
    )

def _analyze_opportunities(topic: str, research: str, provider: str) -> str:
    return _call_llm(
        "You are a business strategist specialising in AI and automation. "
        "Identify exactly 5 concrete growth or competitive opportunities. "
        "Format as a numbered list. Each: bold title + 2-sentence explanation.",
        f"Topic: {topic}\n\nResearch:\n{research}",
        provider, tier="fast",
    )

def _analyze_action_plan(topic: str, research: str, provider: str) -> str:
    return _call_llm(
        "You are a fractional CMO / GTM strategist. Write a 90-day action plan with 3 phases "
        "(Days 1–30, 31–60, 61–90). Each phase: 3 specific, measurable actions. Be concrete.",
        f"Topic: {topic}\n\nResearch:\n{research}",
        provider, tier="fast",
    )

def run_analysis(research: ResearchResult, provider: str) -> AnalysisResult:
    logger.info(f"[Stage 2] Running 3 parallel {provider.upper()} analyses...")
    result = AnalysisResult()
    tasks = {
        "executive_summary": (_analyze_summary,      research.topic, research.raw_content, provider),
        "opportunities":     (_analyze_opportunities, research.topic, research.raw_content, provider),
        "action_plan":       (_analyze_action_plan,   research.topic, research.raw_content, provider),
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn, *args): key for key, (fn, *args) in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                setattr(result, key, future.result())
                logger.info(f"  ✓ {key}")
            except Exception as e:
                logger.error(f"  ✗ {key} failed: {e}")
                result.error = str(e)
    return result

# ---------------------------------------------------------------------------
# Stage 3 — Combine
# ---------------------------------------------------------------------------

def combine_results(topic: str, analysis: AnalysisResult, sources: list[str]) -> str:
    sep = "\n\n" + "─" * 60 + "\n\n"
    parts = [
        f"## Executive Summary\n\n{analysis.executive_summary}",
        f"## Top 5 Opportunities\n\n{analysis.opportunities}",
        f"## 90-Day Action Plan\n\n{analysis.action_plan}",
    ]
    combined = sep.join(parts)
    if sources:
        combined += "\n\n## Sources\n\n" + "\n".join(f"- {s}" for s in sources if s)
    return combined

# ---------------------------------------------------------------------------
# Stage 4 — Format HTML Report (uses powerful tier)
# ---------------------------------------------------------------------------

_HTML_SYSTEM = textwrap.dedent("""
    You are an expert HTML email designer. Convert the Markdown report into a polished,
    self-contained HTML email. Requirements:
    - Inline CSS only (email client compatibility)
    - Max width 680px, centered, clean sans-serif (Arial/Helvetica)
    - Header banner: dark navy (#0d1b2a), white title
    - Section headings in deep blue (#1a3a5c)
    - Numbered opportunity items with light blue left border
    - Action plan phases in subtle grey cards
    - Footer: timestamp + "Generated by Genesis Pipeline"
    - Return ONLY the complete HTML — no markdown, no explanation
""").strip()

def format_html_report(topic: str, combined: str, provider: str) -> str:
    logger.info(f"[Stage 4] Formatting HTML report via {provider.upper()} (powerful tier)...")
    try:
        html = _call_llm(_HTML_SYSTEM, f"Report Topic: {topic}\n\nReport:\n\n{combined}",
                         provider, tier="powerful", max_tokens=4000)
        if not html.strip().lower().startswith(("<!doctype", "<html")):
            html = f"<html><body><pre>{html}</pre></body></html>"
        return html
    except Exception as e:
        logger.error(f"HTML formatting failed: {e}")
        safe = combined.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return f"<html><body><pre style='font-family:Arial;max-width:680px;margin:auto'>{safe}</pre></body></html>"

# ---------------------------------------------------------------------------
# Stage 5 — Send Email
# ---------------------------------------------------------------------------

def send_email(recipient: str, topic: str, html_body: str, dry_run: bool = False) -> bool:
    subject = f"Genesis Pipeline Report: {topic} — {datetime.now().strftime('%Y-%m-%d')}"
    if dry_run or not (SMTP_USER and SMTP_PASS):
        out = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        out.write_text(html_body, encoding="utf-8")
        logger.info(f"[Stage 5] {'DRY-RUN' if dry_run else 'No SMTP'} — saved to {out}")
        return True
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, SMTP_FROM or SMTP_USER, recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo(); s.starttls(); s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_FROM or SMTP_USER, [recipient], msg.as_string())
        logger.info(f"[Stage 5] Email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"[Stage 5] Email failed: {e}")
        return False

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(topic: str, recipient_email: str, provider: str = "anthropic", dry_run: bool = False) -> PipelineRun:
    run = PipelineRun(topic=topic, recipient_email=recipient_email, provider=provider)
    logger.info(f"\n{'='*60}\nGenesis Pipeline — provider={provider.upper()} topic={topic!r}\n{'='*60}\n")

    run.research  = run_research(topic)
    run.analysis  = run_analysis(run.research, provider)
    combined      = combine_results(topic, run.analysis, run.research.sources)
    run.html_report = format_html_report(topic, combined, provider)
    run.sent      = send_email(recipient_email, topic, run.html_report, dry_run=dry_run)

    elapsed = (datetime.now() - run.started_at).total_seconds()
    logger.info(f"\nPipeline complete in {elapsed:.1f}s — provider={provider} sent={run.sent}\n")
    return run

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduled(topic: str, recipient_email: str, provider: str):
    scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
    scheduler.add_job(
        run_pipeline, CronTrigger(day_of_week="fri", hour=6, minute=0, timezone="Europe/Amsterdam"),
        args=[topic, recipient_email, provider], id="genesis_weekly", replace_existing=True,
    )
    logger.info(f"Scheduled every Friday 06:00 Amsterdam | provider={provider}")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genesis Pipeline — multi-provider AI research pipeline")
    parser.add_argument("--topic",    required=True)
    parser.add_argument("--email",    required=True)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "google"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()

    # Validate provider key is present
    key_map = {"anthropic": ANTHROPIC_API_KEY, "openai": OPENAI_API_KEY, "google": GOOGLE_API_KEY}
    if not key_map[args.provider]:
        env_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
        logger.error(f"{env_map[args.provider]} not set in .env — aborting")
        sys.exit(1)

    if args.schedule:
        run_scheduled(args.topic, args.email, args.provider)
    else:
        run_pipeline(args.topic, args.email, args.provider, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
