"""Format and send the daily brand mention digest to Slack."""
from __future__ import annotations
import os
from datetime import date
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from mention_analyzer import MentionResult
from logger import get_logger

log = get_logger("slack_reporter")

SENTIMENT_EMOJI = {"positive": ":green_circle:", "neutral": ":white_circle:",
                   "negative": ":red_circle:", "unknown": ":black_circle:"}


class SlackReporter:
    def __init__(self):
        self._client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        self._channel = os.environ.get("SLACK_CHANNEL", "#brand-monitor")

    def send_digest(self, brand: str, results: list[MentionResult]) -> None:
        blocks = self._build_blocks(brand, results)
        try:
            self._client.chat_postMessage(channel=self._channel, blocks=blocks,
                                          text=f"Brand monitor digest for {brand}")
            log.info("Digest sent to %s", self._channel)
        except SlackApiError as exc:
            log.error("Slack error: %s", exc.response["error"])

    def _build_blocks(self, brand: str, results: list[MentionResult]) -> list[dict]:
        mention_count = sum(1 for r in results if r.mentioned)
        pos = sum(1 for r in results if r.sentiment == "positive")
        neg = sum(1 for r in results if r.sentiment == "negative")

        blocks: list[dict] = [
            {"type": "header", "text": {"type": "plain_text",
             "text": f":mag: Brand Monitor — {brand} — {date.today()}"}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Mentions:* {mention_count}/{len(results)}"},
                {"type": "mrkdwn", "text": f"*Sentiment:* {pos} :green_circle: / {neg} :red_circle:"},
            ]},
            {"type": "divider"},
        ]

        for r in results:
            emoji = SENTIMENT_EMOJI.get(r.sentiment, ":black_circle:")
            status = "Mentioned" if r.mentioned else "Not mentioned"
            excerpt_text = f"\n> _{r.excerpt[:250]}_" if r.excerpt else ""
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"{emoji} *{r.model}* — {status} ({r.sentiment}){excerpt_text}"},
            })

        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": "Powered by LLM Brand Entity Monitor · Saikou.tech"}
        ]})
        return blocks
