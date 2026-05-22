"""Render and send HTML email digest via SMTP."""
from __future__ import annotations
import os
import smtplib
from dataclasses import dataclass
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import NamedTuple
from jinja2 import Environment, BaseLoader
from staleness_scorer import StalenessScore
from logger import get_logger

log = get_logger("email_reporter")

TEMPLATE = """
<!DOCTYPE html>
<html>
<head><style>
  body { font-family: Arial, sans-serif; font-size: 14px; color: #333; }
  h1 { color: #2c3e50; }
  table { border-collapse: collapse; width: 100%; }
  th { background: #2c3e50; color: white; padding: 8px; text-align: left; }
  td { padding: 8px; border-bottom: 1px solid #ddd; }
  .critical { color: #c0392b; font-weight: bold; }
  .high { color: #e67e22; }
  .medium { color: #f1c40f; }
  .low { color: #27ae60; }
</style></head>
<body>
<h1>Content Freshness Report — {{ report_date }}</h1>
<p>Found <strong>{{ total }}</strong> stale posts requiring attention.</p>
<table>
  <tr><th>Priority</th><th>Title</th><th>Days Old</th><th>Word Count</th><th>Recommendation</th></tr>
  {% for item in items %}
  <tr>
    <td class="{{ item.priority }}">{{ item.priority | upper }}</td>
    <td><a href="{{ item.url }}">{{ item.title }}</a></td>
    <td>{{ item.days }}</td>
    <td>{{ item.words or '—' }}</td>
    <td>{{ item.recommendation }}</td>
  </tr>
  {% endfor %}
</table>
<p style="color:#999;font-size:12px;">Content Freshness Guardian · Saikou.tech</p>
</body></html>
"""


@dataclass
class ReportRow:
    priority: str
    title: str
    url: str
    days: int
    words: int
    recommendation: str


class EmailReporter:
    def __init__(self):
        self._smtp_host = os.environ["SMTP_HOST"]
        self._smtp_port = int(os.environ.get("SMTP_PORT", 587))
        self._smtp_user = os.environ["SMTP_USER"]
        self._smtp_pass = os.environ["SMTP_PASS"]
        self._from = os.environ.get("EMAIL_FROM", self._smtp_user)
        self._to = os.environ["EMAIL_TO"]

    def send(self, scored: list[tuple[StalenessScore, str]]) -> None:
        rows = [
            ReportRow(
                priority=s.priority,
                title=s.post.title,
                url=s.post.url,
                days=s.days_since_modified,
                words=s.post.word_count,
                recommendation=rec,
            )
            for s, rec in scored
        ]
        env = Environment(loader=BaseLoader())
        html = env.from_string(TEMPLATE).render(
            report_date=date.today(),
            total=len(rows),
            items=rows,
        )
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Content Freshness Report — {date.today()} — {len(rows)} items"
        msg["From"] = self._from
        msg["To"] = self._to
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._smtp_user, self._smtp_pass)
            server.sendmail(self._from, self._to.split(","), msg.as_string())
        log.info("Email digest sent to %s (%d items)", self._to, len(rows))
