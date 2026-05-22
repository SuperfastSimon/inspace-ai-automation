"""Parse LLM answers and extract brand mention signals."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
from logger import get_logger

log = get_logger("mention_analyzer")


@dataclass
class MentionResult:
    model: str
    brand: str
    mentioned: bool
    sentiment: str          # "positive" | "neutral" | "negative" | "unknown"
    excerpt: Optional[str]  # up to 300 chars around first mention


class MentionAnalyzer:
    SENTIMENT_POSITIVE = {"recommended", "best", "excellent", "great", "top", "leading", "trusted"}
    SENTIMENT_NEGATIVE = {"avoid", "bad", "poor", "worst", "scam", "unreliable", "issues"}

    def analyze(self, model: str, brand: str, answer: str) -> MentionResult:
        lower = answer.lower()
        brand_lower = brand.lower()
        mentioned = brand_lower in lower

        excerpt: Optional[str] = None
        if mentioned:
            idx = lower.index(brand_lower)
            start = max(0, idx - 100)
            end = min(len(answer), idx + 200)
            excerpt = answer[start:end].strip()

        sentiment = self._score_sentiment(lower, brand_lower, mentioned)
        log.debug("brand=%s model=%s mentioned=%s sentiment=%s", brand, model, mentioned, sentiment)
        return MentionResult(model=model, brand=brand, mentioned=mentioned,
                             sentiment=sentiment, excerpt=excerpt)

    def _score_sentiment(self, text: str, brand: str, mentioned: bool) -> str:
        if not mentioned:
            return "unknown"
        window_start = max(0, text.index(brand) - 150)
        window = text[window_start: window_start + 400]
        pos = sum(1 for w in self.SENTIMENT_POSITIVE if w in window)
        neg = sum(1 for w in self.SENTIMENT_NEGATIVE if w in window)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
