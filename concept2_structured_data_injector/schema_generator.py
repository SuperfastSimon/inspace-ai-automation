from __future__ import annotations
import json
import os
import re
from typing import Callable
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from page_analyzer import PageAnalysis
from logger import get_logger

log = get_logger("schema_generator")
_postprocessors: dict[str, Callable[[dict], dict]] = {}

def register_postprocessor(schema_type: str):
    def decorator(fn):
        _postprocessors[schema_type] = fn
        return fn
    return decorator

@register_postprocessor("Article")
def _fix_article(schema: dict) -> dict:
    schema.setdefault("@context", "https://schema.org")
    schema.setdefault("@type", "Article")
    return schema

@register_postprocessor("Product")
def _fix_product(schema: dict) -> dict:
    schema.setdefault("@context", "https://schema.org")
    schema.setdefault("@type", "Product")
    return schema

@register_postprocessor("FAQPage")
def _fix_faq(schema: dict) -> dict:
    schema.setdefault("@context", "https://schema.org")
    schema.setdefault("@type", "FAQPage")
    return schema


class SchemaGenerator:
    SYSTEM = (
        "You are a Schema.org JSON-LD expert. Output ONLY valid raw JSON — "
        "no markdown fences, no explanation, no extra text. "
        "The JSON must be a single Schema.org object with @context and @type."
    )

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def generate(self, page: PageAnalysis) -> dict:
        prompt = (
            f"Page URL: {page.url}\n"
            f"Title: {page.title}\n"
            f"Meta description: {page.meta_description}\n"
            f"Page type: {page.page_type_hint}\n"
            f"Body excerpt: {page.body_text[:1500]}\n\n"
            f"Generate a complete Schema.org JSON-LD object of type {page.page_type_hint}."
        )
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        schema = self._extract_json(raw)
        postprocess = _postprocessors.get(page.page_type_hint, lambda x: x)
        return postprocess(schema)

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"No JSON found in LLM response: {text[:200]}")
