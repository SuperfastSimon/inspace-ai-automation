"""Query multiple LLMs about brand mentions and return raw text answers."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional
import anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import get_logger

log = get_logger("llm_querier")

@dataclass
class LLMResponse:
    model: str
    prompt: str
    answer: str
    error: Optional[str] = None


class LLMQuerier:
    def __init__(self):
        self._anthropic = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _ask_claude(self, prompt: str) -> str:
        msg = self._anthropic.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _ask_gpt(self, prompt: str) -> str:
        resp = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    def query_all(self, prompt: str) -> list[LLMResponse]:
        results: list[LLMResponse] = []
        for label, fn in [("claude-opus-4-7", self._ask_claude), ("gpt-4o-mini", self._ask_gpt)]:
            try:
                answer = fn(prompt)
                results.append(LLMResponse(model=label, prompt=prompt, answer=answer))
                log.info("Got answer from %s (%d chars)", label, len(answer))
            except Exception as exc:
                log.error("Error querying %s: %s", label, exc)
                results.append(LLMResponse(model=label, prompt=prompt, answer="", error=str(exc)))
        return results
