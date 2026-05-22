"""Probe LLMs with category questions and collect which brands appear in answers."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import get_logger

log = get_logger("geo_prober")


@dataclass
class ProbeResult:
    model: str
    query: str
    answer: str
    error: Optional[str] = None


class GEOProber:
    def __init__(self):
        self._anthropic = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _ask_claude(self, query: str) -> str:
        msg = self._anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": query}],
        )
        return msg.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _ask_gpt(self, query: str) -> str:
        resp = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            max_tokens=512,
        )
        return resp.choices[0].message.content

    def probe_all(self, queries: list[str]) -> list[ProbeResult]:
        results: list[ProbeResult] = []
        for query in queries:
            for label, fn in [("claude-haiku", self._ask_claude), ("gpt-4o-mini", self._ask_gpt)]:
                try:
                    answer = fn(query)
                    results.append(ProbeResult(model=label, query=query, answer=answer))
                    log.info("Probe done: model=%s query='%s...'", label, query[:60])
                except Exception as exc:
                    log.error("Probe failed: model=%s error=%s", label, exc)
                    results.append(ProbeResult(model=label, query=query, answer="", error=str(exc)))
        return results
