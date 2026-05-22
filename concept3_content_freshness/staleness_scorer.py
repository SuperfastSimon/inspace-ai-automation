"""Score each post for staleness based on age, word count, and category rules."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from cms_fetcher import PostMeta
from logger import get_logger

log = get_logger("staleness_scorer")

@dataclass
class StalenessScore:
    post: PostMeta
    days_since_modified: int
    score: float           # 0.0 (fresh) – 1.0 (critically stale)
    priority: str          # "critical" | "high" | "medium" | "low"
    reasons: list[str] = field(default_factory=list)


class StalenessScorer:
    # Thresholds in days per priority
    CRITICAL_DAYS = 365
    HIGH_DAYS = 180
    MEDIUM_DAYS = 90

    # Thin content threshold (words)
    THIN_THRESHOLD = 300

    def score(self, post: PostMeta) -> StalenessScore:
        now = datetime.now(timezone.utc)
        delta = now - post.modified.replace(tzinfo=timezone.utc) if post.modified.tzinfo is None \
            else now - post.modified
        days = delta.days
        reasons: list[str] = []
        raw_score = 0.0

        if days >= self.CRITICAL_DAYS:
            raw_score += 0.6
            reasons.append(f"Not updated in {days} days (>{self.CRITICAL_DAYS})")
        elif days >= self.HIGH_DAYS:
            raw_score += 0.4
            reasons.append(f"Not updated in {days} days (>{self.HIGH_DAYS})")
        elif days >= self.MEDIUM_DAYS:
            raw_score += 0.2
            reasons.append(f"Not updated in {days} days (>{self.MEDIUM_DAYS})")

        if post.word_count and post.word_count < self.THIN_THRESHOLD:
            raw_score += 0.2
            reasons.append(f"Thin content ({post.word_count} words)")

        score = min(raw_score, 1.0)

        if score >= 0.6:
            priority = "critical"
        elif score >= 0.4:
            priority = "high"
        elif score >= 0.2:
            priority = "medium"
        else:
            priority = "low"

        return StalenessScore(post=post, days_since_modified=days,
                              score=score, priority=priority, reasons=reasons)
