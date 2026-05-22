from __future__ import annotations
import os
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from opportunity_finder import LinkOpportunity
from logger import get_logger

log = get_logger("anchor_generator")


class AnchorGenerator:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def suggest(self, opp: LinkOpportunity) -> str:
        prompt = (
            f"Source page title: {opp.source.title}\n"
            f"Target page title: {opp.target.title}\n"
            f"Target URL: {opp.target.url}\n\n"
            "Suggest ONE concise anchor text (3-6 words) for an internal link "
            "from the source page to the target page. "
            "Return ONLY the anchor text, nothing else."
        )
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
        )
        anchor = msg.content[0].text.strip().strip('"').strip("'")
        log.debug("Anchor for '%s' → '%s': %s", opp.source.title[:30], opp.target.title[:30], anchor)
        return anchor
