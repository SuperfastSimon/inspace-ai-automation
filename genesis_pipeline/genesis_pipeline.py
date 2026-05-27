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
import html as _html
import ipaddress
import logging
import os
import re
import smtplib
import socket
import ssl
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
from urllib.parse import urlparse

import defusedxml.ElementTree as _ET
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

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

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)

# Reject obviously invalid SMTP config at startup rather than at send time
_VALID_SMTP_HOST = re.fullmatch(r"[A-Za-z0-9.\-]+", SMTP_HOST) is not None
_VALID_SMTP_PORT = 1 <= SMTP_PORT <= 65535

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

ENV_VAR_NAMES = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# ---------------------------------------------------------------------------
# Provider key helper — reads live from env, never stale
# ---------------------------------------------------------------------------

def _get_key(provider: str) -> str:
    return os.getenv(ENV_VAR_NAMES[provider], "")

def _active_providers() -> dict[str, str]:
    return {p: _get_key(p) for p in PROVIDER_MODELS if _get_key(p)}

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
    errors: list[str] = field(default_factory=list)

    @property
    def error(self) -> Optional[str]:
        return self.errors[-1] if self.errors else None

    @error.setter
    def error(self, value: str):
        self.errors.append(value)

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
# Security helpers
# ---------------------------------------------------------------------------

def _strip_header_injection(value: str) -> str:
    """Remove CR/LF characters that could inject extra SMTP headers."""
    return re.sub(r"[\r\n]", "", value)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _validate_email(addr: str) -> bool:
    return bool(_EMAIL_RE.fullmatch(addr))

def _is_private_ip(hostname: str) -> bool:
    """Return True if hostname resolves to a private/loopback address (SSRF guard)."""
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except Exception:
        return False

# Allowed RSS feed hostnames (allowlist — extend as needed)
_RSS_ALLOWED_HOSTS: set[str] = {
    "feeds.feedburner.com",
    "www.wired.com",
    "rss.nytimes.com",
    "feeds.arstechnica.com",
}

def _safe_rss_url(url: str) -> bool:
    """Validate RSS URL against allowlist and private-IP block."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        if host not in _RSS_ALLOWED_HOSTS:
            return False
        if _is_private_ip(host):
            return False
        return True
    except Exception:
        return False

def _escape_slack(text: str) -> str:
    """Escape Slack mrkdwn special characters in untrusted content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------------------------------------------------------------------
# Cost + concurrency limits for run_compare
# ---------------------------------------------------------------------------

MAX_COMPARE_COST_USD = float(os.getenv("MAX_COMPARE_COST_USD", "2.00"))
MAX_COMPARE_PROVIDERS = int(os.getenv("MAX_COMPARE_PROVIDERS", "6"))

# ---------------------------------------------------------------------------
# Provider-agnostic LLM dispatcher
# ---------------------------------------------------------------------------

def _is_auth_error(exc: BaseException) -> bool:
    """Return True for 4xx auth errors that should NOT be retried."""
    msg = str(exc).lower()
    return any(k in msg for k in ("401", "403", "invalid_api_key", "authentication", "unauthorized", "forbidden"))

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception(lambda e: not _is_auth_error(e)),
)
def _call_llm(system: str, user: str, provider: str, tier: str = "fast", max_tokens: int = 1200) -> str:
    model = PROVIDER_MODELS[provider][tier]
    api_key = _get_key(provider)  # always fresh — never stale module-level dict

    # --- Anthropic native SDK ---
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    # --- Google Gemini native SDK ---
    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model_name=model, system_instruction=system)
        return m.generate_content(user).text.strip()

    # --- OpenAI-compatible providers (openai / mistral / groq / openrouter) ---
    else:
        from openai import OpenAI
        base_url = OPENAI_COMPAT_BASES.get(provider)
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
    jina_key = os.getenv("JINA_API_KEY", "")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"
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
                result.errors.append(f"{key}: {e}")  # aggregate all errors
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

def send_email(recipient: str, topic: str, html_body: str, dry_run: bool = False) -> bool:
    # Sanitise header-injection vectors
    safe_topic     = _strip_header_injection(topic)
    safe_recipient = _strip_header_injection(recipient)
    subject = f"Genesis Pipeline Report: {safe_topic} — {datetime.now().strftime('%Y-%m-%d')}"

    if dry_run or not (SMTP_USER and SMTP_PASS):
        out = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        out.write_text(html_body, encoding="utf-8")
        logger.info(f"[Stage 5] {'DRY-RUN' if dry_run else 'No SMTP'} — saved {out}")
        return True

    if not (_VALID_SMTP_HOST and _VALID_SMTP_PORT):
        logger.error(f"[Stage 5] Invalid SMTP config: host={SMTP_HOST!r} port={SMTP_PORT}")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_FROM or SMTP_USER
    msg["To"]      = safe_recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        tls_ctx = ssl.create_default_context()  # verifies server cert, prevents MITM
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls(context=tls_ctx)
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_FROM or SMTP_USER, [safe_recipient], msg.as_string())
        logger.info(f"[Stage 5] Email sent → {safe_recipient}")
        return True
    except Exception as e:
        logger.error(f"[Stage 5] SMTP error: {e}")
        return False

# ---------------------------------------------------------------------------
# Orchestrator  (single definition — Phase 1+2 features built in)
# ---------------------------------------------------------------------------

import sqlite3
import hashlib
import json as _json
import threading

DB_PATH = Path("genesis_runs.db")
_SIDECAR_BASE_DIR = Path(".")  # restrict sidecar writes to this directory

# ---------------------------------------------------------------------------
# SQLite run log
# ---------------------------------------------------------------------------

_db_local = threading.local()

def _db_connect() -> sqlite3.Connection:
    """Return a per-thread SQLite connection (thread-safe, no shared state)."""
    if not getattr(_db_local, "conn", None):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at        TEXT NOT NULL,
                topic         TEXT NOT NULL,
                provider      TEXT NOT NULL,
                recipient     TEXT NOT NULL,
                elapsed_s     REAL,
                input_tokens  INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd      REAL DEFAULT 0.0,
                report_hash   TEXT,
                sent          INTEGER DEFAULT 0,
                dry_run       INTEGER DEFAULT 0,
                sidecar_path  TEXT,
                client        TEXT DEFAULT ''
            )
        """)
        conn.commit()
        _db_local.conn = conn
    return _db_local.conn

def log_run(*, topic: str, provider: str, recipient: str, elapsed_s: float,
            input_tokens: int, output_tokens: int, cost_usd: float,
            report_hash: str, sent: bool, dry_run: bool,
            sidecar_path: str = "", client: str = "") -> int:
    conn = _db_connect()
    cur = conn.execute(
        """INSERT INTO runs
           (run_at, topic, provider, recipient, elapsed_s,
            input_tokens, output_tokens, cost_usd,
            report_hash, sent, dry_run, sidecar_path, client)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().isoformat(), topic, provider, recipient, round(elapsed_s, 2),
         input_tokens, output_tokens, round(cost_usd, 6),
         report_hash, int(sent), int(dry_run), sidecar_path, client)
    )
    conn.commit()
    run_id = cur.lastrowid
    logger.info(f"[DB] Run #{run_id} logged — cost=${cost_usd:.4f} tokens={input_tokens}in/{output_tokens}out")
    return run_id

def get_run_history(limit: int = 10) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _db_connect()
    cur = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

def get_last_report_hash(topic: str, provider: str) -> str:
    """Return hash of most recent run for this topic+provider (for delta mode)."""
    if not DB_PATH.exists():
        return ""
    conn = _db_connect()
    row = conn.execute(
        "SELECT report_hash FROM runs WHERE topic=? AND provider=? ORDER BY id DESC LIMIT 1",
        (topic, provider)
    ).fetchone()
    return row[0] if row else ""

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

COST_PER_1M = {
    "anthropic":  {"input": 3.00,  "output": 15.00},
    "openai":     {"input": 0.15,  "output": 0.60},
    "google":     {"input": 0.075, "output": 0.30},
    "mistral":    {"input": 0.10,  "output": 0.30},
    "groq":       {"input": 0.059, "output": 0.079},
    "openrouter": {"input": 0.10,  "output": 0.30},
}

class CostTracker:
    def __init__(self):
        self._input = 0
        self._output = 0
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
        return f"tokens≈{i}in/{o}out  est_cost≈${self.cost(provider):.4f}"

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# ---------------------------------------------------------------------------
# JSON sidecar
# ---------------------------------------------------------------------------

def write_json_sidecar(*, topic: str, provider: str, recipient: str,
                       analysis: "AnalysisResult", sources: list[str],
                       input_tokens: int, output_tokens: int,
                       cost_usd: float, report_hash: str,
                       run_at: str, filename: str,
                       delta_new: bool = False, client: str = "") -> str:
    # Restrict writes to the base directory — prevent path traversal
    resolved = (_SIDECAR_BASE_DIR / Path(filename).name).resolve()
    if not str(resolved).startswith(str(_SIDECAR_BASE_DIR.resolve())):
        raise ValueError(f"Sidecar path escapes base dir: {filename!r}")

    # Redact PII — store a hash of the recipient address, not the address itself
    recipient_hash = hashlib.sha256(recipient.encode()).hexdigest()[:16]

    data = {
        "schema_version": "1.2",
        "run_at":          run_at,
        "topic":           topic,
        "client":          client,
        "provider":        provider,
        "recipient_hash":  recipient_hash,   # PII redacted
        "delta_new":       delta_new,
        "sources":         sources,
        "analysis": {
            "executive_summary": analysis.executive_summary,
            "opportunities":     analysis.opportunities,
            "action_plan":       analysis.action_plan,
        },
        "cost": {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost_usd, 6),
        },
        "report_hash": report_hash,
    }
    resolved.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[Sidecar] JSON written → {resolved}")
    return str(resolved)

# ---------------------------------------------------------------------------
# Phase 2 — Richer research: RSS feeds + DuckDuckGo HTML fallback
# ---------------------------------------------------------------------------

RSS_FEEDS: list[str] = [
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.wired.com/feed/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
]

def _fetch_rss(topic: str, max_items: int = 5) -> tuple[str, list[str]]:
    """Pull recent RSS headlines matching topic keywords."""
    keywords = set(re.sub(r"[^\w\s]", "", topic.lower()).split())
    items_found: list[tuple[str, str, str]] = []  # (title, link, summary)
    for feed_url in RSS_FEEDS:
        if not _safe_rss_url(feed_url):
            logger.warning(f"[RSS] Skipping untrusted feed URL: {feed_url}")
            continue
        try:
            r = requests.get(feed_url, timeout=10,
                             allow_redirects=False,  # no redirect following
                             headers={"User-Agent": "GenesisResearch/2.0"})
            r.raise_for_status()
            root = _ET.fromstring(r.content)  # defusedxml — XXE-safe
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            # RSS 2.0
            for item in root.iter("item"):
                raw_title   = item.findtext("title") or ""
                raw_link    = item.findtext("link") or ""
                raw_summary = item.findtext("description") or ""
                # HTML-escape untrusted RSS content before embedding in reports
                title   = _html.escape(raw_title.strip())
                link    = raw_link.strip()
                summary = _html.escape(raw_summary[:500].strip())
                text    = (raw_title + " " + raw_summary).lower()
                if any(kw in text for kw in keywords):
                    items_found.append((title, link, summary))
            # Atom
            for entry in root.findall("atom:entry", ns):
                raw_title   = entry.findtext("atom:title", namespaces=ns) or ""
                link_el     = entry.find("atom:link", ns)
                link        = (link_el.get("href", "") if link_el is not None else "")
                raw_summary = entry.findtext("atom:summary", namespaces=ns) or ""
                title   = _html.escape(raw_title.strip())
                summary = _html.escape(raw_summary[:500].strip())
                text    = (raw_title + " " + raw_summary).lower()
                if any(kw in text for kw in keywords):
                    items_found.append((title, link, summary))
        except Exception as e:
            logger.debug(f"[RSS] {feed_url} failed: {e}")
        if len(items_found) >= max_items:
            break

    if not items_found:
        return "", []

    content = "\n\n".join(
        f"[RSS] {title}\n{link}\n{summary}"
        for title, link, summary in items_found[:max_items]
    )
    sources = [link for _, link, _ in items_found[:max_items] if link]
    logger.info(f"[Stage 1] RSS: {len(items_found[:max_items])} relevant items found")
    return content, sources


def _fetch_ddg_fallback(topic: str) -> tuple[str, list[str]]:
    """DuckDuckGo Instant Answer API — zero-auth fallback for extra context."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": topic, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10, headers={"User-Agent": "GenesisResearch/2.0"}
        )
        r.raise_for_status()
        data = r.json()
        abstract     = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        related = [
            f"{t.get('Text', '')[:200]} — {t.get('FirstURL', '')}"
            for t in data.get("RelatedTopics", [])[:4]
            if isinstance(t, dict) and t.get("Text")
        ]
        if not abstract and not related:
            return "", []
        content = ""
        if abstract:
            content += f"[DuckDuckGo Abstract]\n{abstract}\n{abstract_url}\n\n"
        if related:
            content += "[Related Topics]\n" + "\n".join(related)
        sources = [abstract_url] if abstract_url else []
        logger.info("[Stage 1] DuckDuckGo: context retrieved")
        return content.strip(), sources
    except Exception as e:
        logger.debug(f"[DDG] failed: {e}")
        return "", []


def run_research(topic: str) -> ResearchResult:
    """Stage 1 — multi-source research: Jina + RSS + DuckDuckGo."""
    logger.info(f"[Stage 1] Researching: {topic!r} (Jina + RSS + DDG)")
    jina_content, jina_sources = _search_web(topic)
    rss_content,  rss_sources  = _fetch_rss(topic)
    ddg_content,  ddg_sources  = _fetch_ddg_fallback(topic)

    parts   = [p for p in [jina_content, rss_content, ddg_content] if p]
    sources = list(dict.fromkeys(jina_sources + rss_sources + ddg_sources))  # dedup
    combined = "\n\n---\n\n".join(parts) if parts else f"Research topic: {topic}\n(All sources unavailable)"
    logger.info(f"[Stage 1] Total sources: {len(sources)} | content chars: {len(combined)}")
    return ResearchResult(topic=topic, raw_content=combined, sources=sources)

# ---------------------------------------------------------------------------
# Phase 2 — Delta mode: "what's new this week"
# ---------------------------------------------------------------------------

def _is_new_content(html: str, prev_hash: str) -> bool:
    current_hash = hashlib.sha256(html.encode()).hexdigest()[:16]
    return current_hash != prev_hash

def _extract_delta_prompt(prev_hash: str) -> str:
    if not prev_hash:
        return ""
    return (
        "\n\nIMPORTANT: A report on this topic was already sent last week. "
        "Focus ONLY on what is NEW, changed, or emerging since then. "
        "Start your executive summary with 'NEW THIS WEEK:' and skip anything already covered previously."
    )

# ---------------------------------------------------------------------------
# Phase 2 — Multi-tenant YAML config
# ---------------------------------------------------------------------------

CLIENTS_CONFIG_PATH = Path("clients.yaml")

def load_clients_config() -> list[dict]:
    """Load multi-tenant client config from clients.yaml."""
    try:
        import yaml  # optional dep — graceful if missing
    except ImportError:
        logger.warning("[Config] PyYAML not installed — multi-tenant mode unavailable. pip install pyyaml")
        return []
    if not CLIENTS_CONFIG_PATH.exists():
        return []
    with open(CLIENTS_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("clients", [])

def run_all_clients(dry_run: bool = False):
    """Run pipeline for all clients defined in clients.yaml."""
    clients = load_clients_config()
    if not clients:
        logger.error("[MultiTenant] clients.yaml not found or empty. See clients.yaml.example")
        return
    logger.info(f"[MultiTenant] Running {len(clients)} client(s)...")
    for client in clients:
        name      = client.get("name", "unnamed")
        topic     = client.get("topic", "")
        email     = client.get("email", "")
        provider  = client.get("provider", "anthropic")
        if not topic or not email:
            logger.warning(f"[MultiTenant] Skipping {name!r} — missing topic or email")
            continue
        logger.info(f"[MultiTenant] → {name} | {topic!r} | {provider}")
        try:
            run_pipeline(topic, email, provider=provider, dry_run=dry_run, client=client.get("name", ""))
        except Exception as e:
            logger.error(f"[MultiTenant] {name} failed: {e}")

# ---------------------------------------------------------------------------
# Slack webhook delivery
# ---------------------------------------------------------------------------

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

def send_slack(topic: str, provider: str, cost_summary: str,
               executive_summary: str, report_path: str = "",
               client: str = "") -> bool:
    if not SLACK_WEBHOOK_URL:
        return False
    client_tag = f" [{_escape_slack(client)}]" if client else ""
    # Escape LLM/user content to prevent Slack mrkdwn injection
    safe_topic   = _escape_slack(topic)
    safe_summary = _escape_slack(executive_summary[:300])
    safe_path    = _escape_slack(report_path)
    text = (
        f":newspaper: *Genesis Pipeline Report{client_tag}* — _{safe_topic}_\n"
        f">Provider: `{provider}` | {cost_summary}\n"
        f">{safe_summary}{'...' if len(executive_summary) > 300 else ''}"
        + (f"\n>Report: `{safe_path}`" if safe_path else "")
    )
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        logger.info("[Slack] Notification sent ✓")
        return True
    except requests.HTTPError as e:
        logger.warning(f"[Slack] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"[Slack] Delivery failed: {e}")
        return False

# ---------------------------------------------------------------------------
# Orchestrator — single definition, all features
# ---------------------------------------------------------------------------

def run_pipeline(topic: str, recipient_email: str, provider: str = "anthropic",
                 dry_run: bool = False, client: str = "") -> "PipelineRun":
    tracker = CostTracker()
    run_at  = datetime.now().isoformat()
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Input validation
    if not topic.strip():
        raise ValueError("--topic cannot be empty")
    if not _validate_email(recipient_email):
        raise ValueError(f"Invalid email address: {recipient_email!r}")

    run = PipelineRun(topic=topic, recipient_email=recipient_email, provider=provider)
    label = f"[{client}] " if client else ""
    logger.info(f"\n{'='*60}\n{label}Genesis Pipeline | {provider.upper()} | {topic!r}\n{'='*60}\n")

    # Delta: check if we've run this topic before
    prev_hash    = get_last_report_hash(topic, provider)
    delta_prompt = _extract_delta_prompt(prev_hash)

    run.research    = run_research(topic)
    if delta_prompt:
        run.research.raw_content += delta_prompt
    run.analysis    = run_analysis(run.research, provider)
    combined        = combine_results(topic, run.analysis, run.research.sources)
    run.html_report = format_html_report(topic, combined, provider)

    # Token estimates
    in_tok  = _estimate_tokens(run.research.raw_content) * 4
    out_tok = _estimate_tokens(run.analysis.executive_summary +
                               run.analysis.opportunities +
                               run.analysis.action_plan +
                               run.html_report)
    tracker.add(in_tok, out_tok)
    input_tok, output_tok = tracker.totals()
    cost_usd    = tracker.cost(provider)
    cost_str    = tracker.summary(provider)
    report_hash = hashlib.sha256(run.html_report.encode()).hexdigest()[:16]
    delta_new   = _is_new_content(run.html_report, prev_hash)

    # Stage 5 — single delivery point
    if dry_run or not (SMTP_USER and SMTP_PASS):
        out = Path(f"report_{ts}.html")
        out.write_text(run.html_report, encoding="utf-8")
        logger.info(f"[Stage 5] {'DRY-RUN' if dry_run else 'No SMTP'} → saved {out}")
        run.sent = True
    else:
        run.sent = send_email(recipient_email, topic, run.html_report, dry_run=False)

    # JSON sidecar
    sidecar_path = write_json_sidecar(
        topic=topic, provider=provider, recipient=recipient_email,
        analysis=run.analysis, sources=run.research.sources,
        input_tokens=input_tok, output_tokens=output_tok,
        cost_usd=cost_usd, report_hash=report_hash,
        run_at=run_at, filename=f"report_{ts}.json",
        delta_new=delta_new, client=client,
    )

    # SQLite log
    elapsed = (datetime.now() - run.started_at).total_seconds()
    log_run(
        topic=topic, provider=provider, recipient=recipient_email,
        elapsed_s=elapsed, input_tokens=input_tok, output_tokens=output_tok,
        cost_usd=cost_usd, report_hash=report_hash,
        sent=run.sent, dry_run=dry_run, sidecar_path=sidecar_path, client=client,
    )

    # Slack
    send_slack(topic, provider, cost_str, run.analysis.executive_summary,
               report_path=str(Path(f"report_{ts}.html")) if dry_run else "",
               client=client)

    logger.info(f"\nDone in {elapsed:.1f}s | {cost_str} | delta_new={delta_new} | sent={run.sent}\n")
    return run

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduled(topic: str, recipient_email: str, provider: str):
    scheduler = BlockingScheduler(timezone="Europe/Amsterdam")
    scheduler.add_job(run_pipeline, CronTrigger(day_of_week="fri", hour=6, minute=0),
                      args=[topic, recipient_email, provider], id="genesis_weekly", replace_existing=True)
    logger.info(f"Scheduled: every Friday 06:00 Amsterdam | provider={provider}")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")

# ---------------------------------------------------------------------------
# --compare mode (with cost cap + provider cap)
# ---------------------------------------------------------------------------

def run_compare(topic: str, recipient_email: str, dry_run: bool = False):
    active = _active_providers()
    if not active:
        logger.error("No API keys found in .env — set at least one provider key")
        return

    # Enforce provider cap to bound cost amplification
    if len(active) > MAX_COMPARE_PROVIDERS:
        logger.warning(f"[Compare] Capping to {MAX_COMPARE_PROVIDERS} providers (MAX_COMPARE_PROVIDERS)")
        active = dict(list(active.items())[:MAX_COMPARE_PROVIDERS])

    logger.info(f"[Compare] Running {len(active)} providers: {', '.join(active)} | budget cap=${MAX_COMPARE_COST_USD:.2f}")
    research = run_research(topic)

    cost_lock    = threading.Lock()
    running_cost = [0.0]  # mutable container for thread-safe accumulation

    def _run_one(provider: str) -> tuple[str, "AnalysisResult", float]:
        # Pre-flight cost estimate using fast-tier rates
        est_in  = _estimate_tokens(research.raw_content) * 4
        est_out = 2000
        rates   = COST_PER_1M.get(provider, {"input": 0.10, "output": 0.30})
        est_cost = est_in / 1_000_000 * rates["input"] + est_out / 1_000_000 * rates["output"]
        with cost_lock:
            if running_cost[0] + est_cost > MAX_COMPARE_COST_USD:
                raise RuntimeError(
                    f"Budget cap ${MAX_COMPARE_COST_USD:.2f} would be exceeded; skipping {provider}"
                )
            running_cost[0] += est_cost

        t0       = datetime.now()
        analysis = run_analysis(research, provider)
        combined = combine_results(topic, analysis, research.sources)
        html     = format_html_report(topic, combined, provider)
        elapsed  = (datetime.now() - t0).total_seconds()
        return html, analysis, elapsed

    results: dict[str, tuple[str, "AnalysisResult", float]] = {}
    with ThreadPoolExecutor(max_workers=min(len(active), MAX_COMPARE_PROVIDERS)) as pool:
        futures = {pool.submit(_run_one, p): p for p in active}
        for future in as_completed(futures):
            p = futures[future]
            try:
                results[p] = future.result()
                logger.info(f"  ✓ {p}")
            except Exception as e:
                logger.error(f"  ✗ {p}: {e}")

    html = _build_comparison_html(topic, results, research.sources)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = Path(f"compare_{ts}.html")
    out.write_text(html, encoding="utf-8")
    logger.info(f"[Compare] Report saved → {out}")

    if not dry_run and SMTP_USER and SMTP_PASS:
        send_email(recipient_email, f"[Compare] {topic}", html, dry_run=False)


def _build_comparison_html(topic: str, results: dict, sources: list[str]) -> str:
    colors = ["#0f3460", "#1a6b3c", "#7c2d12", "#1e3a5f", "#4a1d6e", "#0e4b3a"]
    provider_colors = {p: colors[i % len(colors)] for i, p in enumerate(results)}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    headers_html = "".join(
        f'<th style="background:{provider_colors[p]};color:#fff;padding:12px 16px;'
        f'text-align:left;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px">'
        f'{p}</th>'
        for p in results
    )

    def _section_row(label: str, attr: str) -> str:
        cells = "".join(
            f'<td style="padding:14px 16px;vertical-align:top;border-bottom:1px solid #e5e7eb;'
            f'font-size:0.82rem;line-height:1.6;border-left:3px solid {provider_colors[p]}">'
            f'{getattr(analysis, attr, "—").replace(chr(10), "<br>")}</td>'
            for p, (_, analysis, _) in results.items()
        )
        return (
            f'<tr><td colspan="{len(results)+1}" style="background:#f8fafc;'
            f'padding:8px 16px;font-weight:700;color:#374151;'
            f'font-size:0.78rem;text-transform:uppercase;letter-spacing:1px">'
            f'{label}</td></tr>'
            f'<tr>{cells}</tr>'
        )

    perf_row = "".join(
        f'<td style="padding:10px 16px;font-size:0.8rem;color:#6b7280">'
        f'⏱ {elapsed:.1f}s</td>'
        for _, (_, _, elapsed) in results.items()
    )

    sources_html = "".join(
        f'<li style="font-size:0.78rem;margin:3px 0"><a href="{s}" style="color:#3b82f6">{s}</a></li>'
        for s in sources if s
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Genesis Compare: {topic}</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f1f5f9;color:#1e293b}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px}}
.hero{{background:linear-gradient(135deg,#0d1b2a,#1a3a5c);color:#fff;padding:32px;border-radius:12px;margin-bottom:24px}}
.hero h1{{font-size:1.6rem;font-weight:800}}.hero p{{color:#94a3b8;margin-top:6px;font-size:0.9rem}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
</style></head>
<body><div class="wrap">
<div class="hero">
  <h1>Genesis Pipeline — Provider Comparison</h1>
  <p>Topic: {topic} &nbsp;|&nbsp; {len(results)} providers &nbsp;|&nbsp; {ts}</p>
</div>
<table>
<thead><tr><th style="background:#0d1b2a;color:#fff;padding:12px 16px;text-align:left;width:120px">Provider</th>{headers_html}</tr></thead>
<tbody>
{_section_row("Executive Summary", "executive_summary")}
{_section_row("Top Opportunities", "opportunities")}
{_section_row("90-Day Action Plan", "action_plan")}
<tr><td style="padding:10px 16px;font-size:0.8rem;font-weight:600;color:#374151">Speed</td>{perf_row}</tr>
</tbody></table>
<div style="margin-top:20px;background:#fff;border-radius:8px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
  <p style="font-weight:700;font-size:0.85rem;margin-bottom:8px">Research Sources</p>
  <ul style="list-style:none;padding:0">{sources_html}</ul>
</div>
</div></body></html>"""

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

        Phase 2 features:
          --clients   Run all clients defined in clients.yaml
          --compare   Side-by-side HTML report across all configured providers
          --delta     Highlight only new content vs last run (auto-enabled when history exists)
        """)
    )
    parser.add_argument("--topic",    default="")
    parser.add_argument("--email",    default="")
    parser.add_argument("--provider", default="anthropic", choices=PROVIDERS)
    parser.add_argument("--dry-run",  action="store_true", help="Save HTML locally, skip email")
    parser.add_argument("--schedule", action="store_true", help="Run weekly Friday 06:00 Amsterdam")
    parser.add_argument("--compare",  action="store_true", help="All providers, side-by-side report")
    parser.add_argument("--clients",  action="store_true", help="Run all clients from clients.yaml")
    parser.add_argument("--history",  action="store_true", help="Print last 10 runs from DB")
    args = parser.parse_args()

    if args.history:
        runs = get_run_history(10)
        if not runs:
            print("No runs recorded yet.")
        for r in runs:
            print(f"#{r['id']} {r['run_at'][:16]} | {r['provider']:10} | {r['topic'][:40]:40} | "
                  f"${r['cost_usd']:.4f} | sent={'yes' if r['sent'] else 'no'}")
        return

    if args.clients:
        run_all_clients(dry_run=args.dry_run)
        return

    if not args.topic:
        parser.error("--topic is required (unless --clients or --history)")
    if not args.email and not args.compare:
        parser.error("--email is required")

    if not args.compare and not _get_key(args.provider):
        logger.error(f"{ENV_VAR_NAMES[args.provider]} not set in .env")
        sys.exit(1)

    if args.compare:
        run_compare(args.topic, args.email or "noreply@genesis", dry_run=args.dry_run)
    elif args.schedule:
        run_scheduled(args.topic, args.email, args.provider)
    else:
        run_pipeline(args.topic, args.email, args.provider, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
