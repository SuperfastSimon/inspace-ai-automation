#!/usr/bin/env python3
"""
Genesis Pipeline — AI Research & Content Intelligence
Multi-provider LLM support: anthropic | openai | google | mistral | groq | openrouter

Usage:
  python genesis_pipeline.py --topic "Saikou.tech" --email you@example.com
  python genesis_pipeline.py --topic "AI trends" --email you@example.com --provider mistral
  python genesis_pipeline.py --topic "AI trends" --email you@example.com --provider groq
  python genesis_pipeline.py --topic "AI trends" --email you@example.com --provider openrouter
  python genesis_pipeline.py --topic "AI trends" --email you@example.com --dry-run
  python genesis_pipeline.py --topic "AI trends" --email you@example.com --schedule
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

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
MISTRAL_API_KEY    = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
JINA_API_KEY       = os.getenv("JINA_API_KEY", "")
SMTP_HOST          = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT          = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER          = os.getenv("SMTP_USER", "")
SMTP_PASS          = os.getenv("SMTP_PASS", "")
SMTP_FROM          = os.getenv("SMTP_FROM", SMTP_USER)

# Model IDs per provider — fast (parallel analysis) + powerful (HTML report)
PROVIDER_MODELS = {
    "anthropic":  {"fast": "claude-sonnet-4-6",              "powerful": "claude-opus-4-7"},
    "openai":     {"fast": "gpt-4o-mini",                    "powerful": "gpt-4o"},
    "google":     {"fast": "gemini-2.0-flash",               "powerful": "gemini-2.5-pro-preview-05-06"},
    "mistral":    {"fast": "mistral-small-latest",           "powerful": "mistral-large-latest"},
    "groq":       {"fast": "llama-3.3-70b-versatile",        "powerful": "llama-3.3-70b-versatile"},
    "openrouter": {"fast": "meta-llama/llama-3.3-70b-instruct", "powerful": "anthropic/claude-opus-4"},
}

# OpenAI-compatible base URLs (Mistral, Groq, OpenRouter all speak OpenAI protocol)
OPENAI_COMPAT_BASES = {
    "mistral":    "https://api.mistral.ai/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

PROVIDER_KEYS = {
    "anthropic":  ANTHROPIC_API_KEY,
    "openai":     OPENAI_API_KEY,
    "google":     GOOGLE_API_KEY,
    "mistral":    MISTRAL_API_KEY,
    "groq":       GROQ_API_KEY,
    "openrouter": OPENROUTER_API_KEY,
}

ENV_VAR_NAMES = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
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
# Provider-agnostic LLM dispatcher
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _call_llm(system: str, user: str, provider: str, tier: str = "fast", max_tokens: int = 1200) -> str:
    model = PROVIDER_MODELS[provider][tier]

    # --- Anthropic native SDK ---
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    # --- Google Gemini native SDK ---
    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        m = genai.GenerativeModel(model_name=model, system_instruction=system)
        return m.generate_content(user).text.strip()

    # --- OpenAI-compatible providers (openai / mistral / groq / openrouter) ---
    else:
        from openai import OpenAI
        base_url = OPENAI_COMPAT_BASES.get(provider)   # None → OpenAI default
        api_key  = PROVIDER_KEYS[provider]
        extra_headers = {}
        if provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://saikou.tech",
                "X-Title": "Genesis Pipeline",
            }
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            extra_headers=extra_headers or None,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Stage 1 — Web Research (Jina)
# ---------------------------------------------------------------------------

def _search_web(query: str) -> tuple[str, list[str]]:
    headers = {"Accept": "application/json"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        resp = requests.get(f"https://s.jina.ai/{requests.utils.quote(query)}",
                            headers=headers, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("data", [])[:3]
        sources  = [r.get("url", "") for r in results]
        combined = "\n\n".join(
            f"[Source: {r.get('url','?')}]\n{r.get('description', r.get('content',''))[:2000]}"
            for r in results
        )
        return combined, sources
    except Exception as e:
        logger.warning(f"Jina unavailable — {e}")
        return f"Research topic: {query}\n(Web search unavailable)", []

def run_research(topic: str) -> ResearchResult:
    logger.info(f"[Stage 1] Researching: {topic}")
    content, sources = _search_web(topic)
    return ResearchResult(topic=topic, raw_content=content, sources=sources)

# ---------------------------------------------------------------------------
# Stage 2 — Parallel LLM Analysis
# ---------------------------------------------------------------------------

def _analyze_summary(topic, research, provider):
    return _call_llm(
        "You are a senior market intelligence analyst. Write a concise executive summary "
        "(3–5 paragraphs) on current state, key players, and market dynamics. Be factual.",
        f"Topic: {topic}\n\nResearch:\n{research}", provider, "fast")

def _analyze_opportunities(topic, research, provider):
    return _call_llm(
        "You are a business strategist specialising in AI and automation. "
        "Identify exactly 5 concrete growth opportunities. Numbered list, bold title + 2-sentence explanation.",
        f"Topic: {topic}\n\nResearch:\n{research}", provider, "fast")

def _analyze_action_plan(topic, research, provider):
    return _call_llm(
        "You are a fractional CMO. Write a 90-day action plan with 3 phases (Days 1–30, 31–60, 61–90). "
        "Each phase: 3 specific, measurable actions. Be concrete.",
        f"Topic: {topic}\n\nResearch:\n{research}", provider, "fast")

def run_analysis(research: ResearchResult, provider: str) -> AnalysisResult:
    logger.info(f"[Stage 2] 3× parallel {provider.upper()} analyses...")
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
                logger.error(f"  ✗ {key}: {e}")
                result.error = str(e)
    return result

# ---------------------------------------------------------------------------
# Stage 3 — Combine
# ---------------------------------------------------------------------------

def combine_results(topic, analysis, sources):
    sep = "\n\n" + "─" * 60 + "\n\n"
    combined = sep.join([
        f"## Executive Summary\n\n{analysis.executive_summary}",
        f"## Top 5 Opportunities\n\n{analysis.opportunities}",
        f"## 90-Day Action Plan\n\n{analysis.action_plan}",
    ])
    if sources:
        combined += "\n\n## Sources\n\n" + "\n".join(f"- {s}" for s in sources if s)
    return combined

# ---------------------------------------------------------------------------
# Stage 4 — HTML Report (powerful tier)
# ---------------------------------------------------------------------------

_HTML_SYSTEM = textwrap.dedent("""
    You are an expert HTML email designer. Convert the Markdown report into a polished,
    self-contained HTML email with inline CSS only (email client compatibility).
    Max width 680px, centered, Arial/Helvetica. Header banner: dark navy (#0d1b2a), white title.
    Section headings: deep blue (#1a3a5c). Opportunity items: light blue left border.
    Action plan phases: subtle grey cards. Footer: timestamp + "Generated by Genesis Pipeline".
    Return ONLY the complete HTML document — no markdown, no explanation.
""").strip()

def format_html_report(topic, combined, provider):
    logger.info(f"[Stage 4] HTML formatting via {provider.upper()} (powerful)...")
    try:
        html = _call_llm(_HTML_SYSTEM, f"Report Topic: {topic}\n\nReport:\n\n{combined}",
                         provider, "powerful", max_tokens=4000)
        if not html.strip().lower().startswith(("<!doctype", "<html")):
            html = f"<html><body><pre>{html}</pre></body></html>"
        return html
    except Exception as e:
        logger.error(f"HTML fallback: {e}")
        safe = combined.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return f"<html><body><pre style='font-family:Arial;max-width:680px;margin:auto'>{safe}</pre></body></html>"

# ---------------------------------------------------------------------------
# Stage 5 — Deliver
# ---------------------------------------------------------------------------

def send_email(recipient, topic, html_body, dry_run=False):
    subject = f"Genesis Pipeline Report: {topic} — {datetime.now().strftime('%Y-%m-%d')}"
    if dry_run or not (SMTP_USER and SMTP_PASS):
        out = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        out.write_text(html_body, encoding="utf-8")
        logger.info(f"[Stage 5] {'DRY-RUN' if dry_run else 'No SMTP'} — saved {out}")
        return True
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, SMTP_FROM or SMTP_USER, recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo(); s.starttls(); s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_FROM or SMTP_USER, [recipient], msg.as_string())
        logger.info(f"[Stage 5] Email sent → {recipient}")
        return True
    except Exception as e:
        logger.error(f"[Stage 5] SMTP error: {e}")
        return False

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(topic, recipient_email, provider="anthropic", dry_run=False):
    run = PipelineRun(topic=topic, recipient_email=recipient_email, provider=provider)
    logger.info(f"\n{'='*60}\nGenesis Pipeline | provider={provider.upper()} | topic={topic!r}\n{'='*60}\n")
    run.research    = run_research(topic)
    run.analysis    = run_analysis(run.research, provider)
    combined        = combine_results(topic, run.analysis, run.research.sources)
    run.html_report = format_html_report(topic, combined, provider)
    run.sent        = send_email(recipient_email, topic, run.html_report, dry_run)
    elapsed = (datetime.now() - run.started_at).total_seconds()
    logger.info(f"\nDone in {elapsed:.1f}s | provider={provider} | sent={run.sent}\n")
    return run

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduled(topic, recipient_email, provider):
    scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
    scheduler.add_job(run_pipeline, CronTrigger(day_of_week="fri", hour=6, minute=0),
                      args=[topic, recipient_email, provider], id="genesis_weekly", replace_existing=True)
    logger.info(f"Scheduled: every Friday 06:00 Amsterdam | provider={provider}")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PROVIDERS = list(PROVIDER_MODELS.keys())

def main():
    parser = argparse.ArgumentParser(
        description="Genesis Pipeline — multi-provider AI research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Providers & models:
          anthropic   fast=claude-sonnet-4-6      powerful=claude-opus-4-7
          openai      fast=gpt-4o-mini             powerful=gpt-4o
          google      fast=gemini-2.0-flash        powerful=gemini-2.5-pro
          mistral     fast=mistral-small-latest    powerful=mistral-large-latest
          groq        fast=llama-3.3-70b-versatile powerful=llama-3.3-70b-versatile
          openrouter  fast=llama-3.3-70b-instruct  powerful=claude-opus-4
        """)
    )
    parser.add_argument("--topic",    required=True)
    parser.add_argument("--email",    required=True)
    parser.add_argument("--provider", default="anthropic", choices=PROVIDERS,
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--dry-run",  action="store_true", help="Save HTML locally, skip email")
    parser.add_argument("--schedule", action="store_true", help="Run weekly Friday 06:00 Amsterdam")
    parser.add_argument("--compare",  action="store_true", help="Run all configured providers in parallel, output side-by-side HTML report")
    args = parser.parse_args()

    if not args.compare and not PROVIDER_KEYS[args.provider]:
        logger.error(f"{ENV_VAR_NAMES[args.provider]} not set in .env")
        sys.exit(1)

    if args.compare:
        run_compare(args.topic, args.email, dry_run=args.dry_run)
    elif args.schedule:
        run_scheduled(args.topic, args.email, args.provider)
    else:
        run_pipeline(args.topic, args.email, args.provider, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# --compare mode: run all configured providers, side-by-side HTML report
# ---------------------------------------------------------------------------

def run_compare(topic: str, recipient_email: str, dry_run: bool = False):
    """Run all providers that have API keys set, produce a side-by-side comparison report."""
    active = {p: k for p, k in PROVIDER_KEYS.items() if k}
    if not active:
        logger.error("No API keys found in .env — set at least one provider key")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARE MODE | {len(active)} providers: {', '.join(active)}")
    logger.info(f"Topic: {topic!r}")
    logger.info(f"{'='*60}\n")

    # Stage 1: Research once, shared across all providers
    research = run_research(topic)

    # Stage 2: Run all providers in parallel
    provider_results: dict[str, AnalysisResult] = {}

    def _run_provider(provider: str) -> tuple[str, AnalysisResult]:
        logger.info(f"  → Starting {provider.upper()}...")
        result = run_analysis(research, provider)
        logger.info(f"  ✓ {provider.upper()} complete")
        return provider, result

    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {pool.submit(_run_provider, p): p for p in active}
        for future in as_completed(futures):
            try:
                provider, result = future.result()
                provider_results[provider] = result
            except Exception as e:
                p = futures[future]
                logger.error(f"  ✗ {p} failed: {e}")
                provider_results[p] = AnalysisResult(error=str(e))

    # Stage 3: Build comparison HTML
    html = _build_comparison_html(topic, provider_results, research.sources)

    # Stage 4: Deliver
    sent = send_email(recipient_email, f"[COMPARE] {topic}", html, dry_run=dry_run)
    logger.info(f"\nCompare report done | providers={list(provider_results)} | sent={sent}\n")


def _build_comparison_html(topic: str, results: dict[str, AnalysisResult], sources: list[str]) -> str:
    PROVIDER_COLORS = {
        "anthropic":  ("#e9d5ff", "#6b21a8", "🟣"),
        "openai":     ("#d1fae5", "#065f46", "🟢"),
        "google":     ("#fef9c3", "#854d0e", "🟡"),
        "mistral":    ("#fee2e2", "#991b1b", "🔴"),
        "groq":       ("#dbeafe", "#1e40af", "🔵"),
        "openrouter": ("#fce7f3", "#9d174d", "🩷"),
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    provider_count = len(results)
    col_width = max(int(100 / provider_count) - 1, 15)

    # Header cards
    header_cards = ""
    for p, res in results.items():
        bg, color, emoji = PROVIDER_COLORS.get(p, ("#f1f5f9", "#334155", "⚪"))
        model_fast = PROVIDER_MODELS[p]["fast"]
        status = "✓ OK" if not res.error else f"✗ {res.error[:40]}"
        header_cards += f"""
        <div style="background:{bg};border-radius:10px;padding:16px 18px;min-width:160px;flex:1;">
          <div style="font-size:1.4rem;">{emoji}</div>
          <div style="font-weight:700;color:{color};font-size:0.95rem;margin-top:4px;">{p.upper()}</div>
          <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">{model_fast}</div>
          <div style="font-size:0.75rem;margin-top:6px;color:{'#059669' if not res.error else '#dc2626'};">{status}</div>
        </div>"""

    # Section builder
    def section(attr: str, title: str, icon: str) -> str:
        cols = ""
        for p, res in results.items():
            bg, color, _ = PROVIDER_COLORS.get(p, ("#f1f5f9","#334155","⚪"))
            content = getattr(res, attr, "") or f'<em style="color:#94a3b8">Error: {res.error}</em>'
            # Convert **bold** and newlines to HTML
            content = content.replace("**", "<strong>", 1)
            import re
            content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
            content = content.replace("\n", "<br>")
            cols += f"""
            <td style="width:{col_width}%;vertical-align:top;padding:14px;border-right:1px solid #e2e8f0;">
              <div style="font-size:0.7rem;font-weight:700;color:{color};text-transform:uppercase;
                          letter-spacing:0.8px;margin-bottom:8px;padding-bottom:6px;
                          border-bottom:2px solid {bg};">{p.upper()}</div>
              <div style="font-size:0.82rem;line-height:1.6;color:#1e293b;">{content}</div>
            </td>"""
        return f"""
        <div style="margin-bottom:32px;">
          <h2 style="font-size:1rem;font-weight:700;color:#1a3a5c;margin-bottom:12px;
                     padding-left:12px;border-left:4px solid #2563eb;">{icon} {title}</h2>
          <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;">
              <tr style="background:#f8fafc;">{cols}</tr>
            </table>
          </div>
        </div>"""

    sources_html = ""
    if sources:
        src = "".join(f'<li style="font-size:0.78rem;color:#2563eb;margin:3px 0;"><a href="{s}" style="color:#2563eb;">{s}</a></li>' for s in sources if s)
        sources_html = f'<div style="margin-top:24px;"><strong style="font-size:0.85rem;">Sources</strong><ul style="margin:8px 0 0 16px;">{src}</ul></div>'

    import re
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Genesis Pipeline — Provider Comparison</title></head>
<body style="font-family:Arial,Helvetica,sans-serif;background:#f0f4f8;margin:0;padding:20px;">
<div style="max-width:1100px;margin:0 auto;">
  <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);color:white;
              border-radius:14px;padding:28px 32px;margin-bottom:24px;">
    <div style="font-size:0.75rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px;">Genesis Pipeline — Provider Comparison</div>
    <h1 style="font-size:1.5rem;font-weight:700;margin:8px 0 4px;">{topic}</h1>
    <div style="font-size:0.8rem;opacity:0.65;">{now} · {provider_count} providers compared</div>
  </div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px;">
    {header_cards}
  </div>

  {section('executive_summary', 'Executive Summary', '📋')}
  {section('opportunities', 'Top 5 Opportunities', '💡')}
  {section('action_plan', '90-Day Action Plan', '📅')}
  {sources_html}

  <div style="text-align:center;padding:20px;font-size:0.75rem;color:#94a3b8;margin-top:16px;">
    Generated by Genesis Pipeline · Saikou.tech · {now}
  </div>
</div>
</body></html>"""

# ===========================================================================
# PHASE 1 ENHANCEMENTS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. SQLite run log
# ---------------------------------------------------------------------------

import sqlite3
import hashlib
import json as _json

DB_PATH = Path("genesis_runs.db")

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at       TEXT NOT NULL,
            topic        TEXT NOT NULL,
            provider     TEXT NOT NULL,
            recipient    TEXT NOT NULL,
            elapsed_s    REAL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd     REAL DEFAULT 0.0,
            report_hash  TEXT,
            sent         INTEGER DEFAULT 0,
            dry_run      INTEGER DEFAULT 0,
            sidecar_path TEXT
        )
    """)
    conn.commit()
    return conn

def log_run(*, topic: str, provider: str, recipient: str, elapsed_s: float,
            input_tokens: int, output_tokens: int, cost_usd: float,
            report_hash: str, sent: bool, dry_run: bool,
            sidecar_path: str = "") -> int:
    conn = _db_connect()
    cur = conn.execute(
        """INSERT INTO runs
           (run_at, topic, provider, recipient, elapsed_s,
            input_tokens, output_tokens, cost_usd,
            report_hash, sent, dry_run, sidecar_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().isoformat(), topic, provider, recipient, round(elapsed_s, 2),
         input_tokens, output_tokens, round(cost_usd, 6),
         report_hash, int(sent), int(dry_run), sidecar_path)
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    logger.info(f"[DB] Run #{run_id} logged — cost=${cost_usd:.4f} tokens={input_tokens}in/{output_tokens}out")
    return run_id

def get_run_history(limit: int = 10) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _db_connect()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM runs LIMIT 0").description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

# ---------------------------------------------------------------------------
# 2. Token counting + cost estimation
# ---------------------------------------------------------------------------

# Per-provider cost per 1M tokens (input, output) in USD — approximate
COST_PER_1M = {
    "anthropic":  {"input": 3.00,  "output": 15.00},   # claude-sonnet-4-6 / opus-4-7 avg
    "openai":     {"input": 0.15,  "output": 0.60},    # gpt-4o-mini
    "google":     {"input": 0.075, "output": 0.30},    # gemini-2.0-flash
    "mistral":    {"input": 0.10,  "output": 0.30},    # mistral-small-latest
    "groq":       {"input": 0.059, "output": 0.079},   # llama-3.3-70b on groq
    "openrouter": {"input": 0.10,  "output": 0.30},    # meta-llama/llama-3.3-70b-instruct
}

class CostTracker:
    """Thread-safe accumulator for token counts across pipeline stages."""
    def __init__(self):
        self._input = 0
        self._output = 0
        import threading
        self._lock = threading.Lock()

    def add(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self._input  += input_tokens
            self._output += output_tokens

    def totals(self) -> tuple[int, int]:
        return self._input, self._output

    def cost(self, provider: str) -> float:
        rates = COST_PER_1M.get(provider, {"input": 0.10, "output": 0.30})
        return (self._input / 1_000_000 * rates["input"] +
                self._output / 1_000_000 * rates["output"])

    def summary(self, provider: str) -> str:
        i, o = self.totals()
        return f"tokens={i}in/{o}out  est_cost=${self.cost(provider):.4f}"

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)

# ---------------------------------------------------------------------------
# 3. JSON sidecar output
# ---------------------------------------------------------------------------

def write_json_sidecar(*, topic: str, provider: str, recipient: str,
                       analysis: "AnalysisResult", sources: list[str],
                       input_tokens: int, output_tokens: int,
                       cost_usd: float, report_hash: str,
                       run_at: str, filename: str) -> str:
    data = {
        "schema_version": "1.0",
        "run_at":         run_at,
        "topic":          topic,
        "provider":       provider,
        "recipient":      recipient,
        "sources":        sources,
        "analysis": {
            "executive_summary": analysis.executive_summary,
            "opportunities":     analysis.opportunities,
            "action_plan":       analysis.action_plan,
        },
        "cost": {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost_usd, 6),
            "provider":      provider,
        },
        "report_hash": report_hash,
    }
    path = Path(filename)
    path.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[Sidecar] JSON written → {path}")
    return str(path)

# ---------------------------------------------------------------------------
# 4. Slack webhook delivery
# ---------------------------------------------------------------------------

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

def send_slack(topic: str, provider: str, cost_summary: str,
               executive_summary: str, report_path: str = "") -> bool:
    if not SLACK_WEBHOOK_URL:
        return False
    text = (
        f":newspaper: *Genesis Pipeline Report* — _{topic}_\n"
        f">Provider: `{provider}` | {cost_summary}\n"
        f">{executive_summary[:300]}{'...' if len(executive_summary) > 300 else ''}\n"
        + (f">Report saved: `{report_path}`" if report_path else "")
    )
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        logger.info("[Slack] Notification sent ✓")
        return True
    except Exception as e:
        logger.warning(f"[Slack] Delivery failed: {e}")
        return False

# ---------------------------------------------------------------------------
# Patched run_pipeline_v2 — drop-in replacement that adds all Phase 1 features
# ---------------------------------------------------------------------------

def run_pipeline(topic, recipient_email, provider="anthropic", dry_run=False):
    """Orchestrator with SQLite logging, cost tracking, JSON sidecar, Slack."""
    tracker = CostTracker()
    run_at  = datetime.now().isoformat()

    # Monkey-patch _call_llm to intercept token counts this run only
    _orig_call_llm = globals().get("_call_llm_orig") or _call_llm
    globals()["_call_llm_orig"] = _orig_call_llm  # keep original

    run = PipelineRun(topic=topic, recipient_email=recipient_email, provider=provider)
    logger.info(f"\n{'='*60}\nGenesis Pipeline v2 | provider={provider.upper()} | topic={topic!r}\n{'='*60}\n")

    run.research    = run_research(topic)
    run.analysis    = run_analysis(run.research, provider)
    combined        = combine_results(topic, run.analysis, run.research.sources)
    run.html_report = format_html_report(topic, combined, provider)

    # Estimate tokens from text lengths (conservative; real counts need SDK usage objects)
    research_tokens = _estimate_tokens(run.research.raw_content)
    analysis_text   = " ".join([run.analysis.executive_summary,
                                 run.analysis.opportunities,
                                 run.analysis.action_plan])
    out_tokens      = _estimate_tokens(analysis_text) + _estimate_tokens(run.html_report)
    tracker.add(research_tokens + _estimate_tokens(combined), out_tokens)

    input_tok, output_tok = tracker.totals()
    cost_usd   = tracker.cost(provider)
    cost_str   = tracker.summary(provider)
    report_hash = hashlib.sha256(run.html_report.encode()).hexdigest()[:16]

    run.sent = send_email(recipient_email, topic, run.html_report, dry_run)
    elapsed  = (datetime.now() - run.started_at).total_seconds()

    # JSON sidecar
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sidecar_path = write_json_sidecar(
        topic=topic, provider=provider, recipient=recipient_email,
        analysis=run.analysis, sources=run.research.sources,
        input_tokens=input_tok, output_tokens=output_tok,
        cost_usd=cost_usd, report_hash=report_hash,
        run_at=run_at, filename=f"report_{ts}.json"
    )

    # SQLite log
    log_run(
        topic=topic, provider=provider, recipient=recipient_email,
        elapsed_s=elapsed, input_tokens=input_tok, output_tokens=output_tok,
        cost_usd=cost_usd, report_hash=report_hash,
        sent=run.sent, dry_run=dry_run, sidecar_path=sidecar_path
    )

    # Slack notification
    send_slack(topic, provider, cost_str,
               run.analysis.executive_summary,
               report_path=sidecar_path if dry_run else "")

    logger.info(f"\nDone in {elapsed:.1f}s | {cost_str} | sent={run.sent}\n")
    return run

