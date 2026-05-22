"""Use Claude to produce a 1-sentence refresh recommendation per stale post."""
from __future__ import annotations
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic
from staleness_scorer import StalenessScore
from logger import get_logger

log = get_logger("ai_summarizer")


class AISummarizer:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def recommend(self, item: StalenessScore) -> str:
        prompt = (
            f"Post title: {item.post.title}\n"
            f"Last modified: {item.post.modified.date()}\n"
            f"Days since update: {item.days_since_modified}\n"
            f"Word count: {item.post.word_count or 'unknown'}\n"
            f"Staleness reasons: {'; '.join(item.reasons)}\n\n"
            "Write ONE concise sentence (max 25 words) recommending what content update action "
            "a content manager should take. Be specific and actionable."
        )
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        rec = msg.content[0].text.strip()
        log.debug("Recommendation for '%s': %s", item.post.title[:50], rec)
        return rec
