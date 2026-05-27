# Privacy Policy

**Last updated:** May 2026

## What data this pipeline processes

| Data | Purpose | Stored? |
|------|---------|---------|
| Research topic (string) | LLM prompt input | Yes — in SQLite run log |
| Recipient email address | SMTP delivery only | Yes — in SQLite run log |
| Web research content | LLM context | No — in-memory only |
| LLM-generated report | Email delivery | Yes — as local HTML file (dry-run) |
| Report content hash (16 chars) | Deduplication / delta mode | Yes — in SQLite run log |
| API keys | Provider authentication | No — env vars only, never persisted |
| Run cost estimate | Usage tracking | Yes — in SQLite run log |

## Third-party services used

| Service | Data sent | Policy |
|---------|-----------|--------|
| Jina Reader API | Search query (topic) | [jina.ai/privacy](https://jina.ai/privacy) |
| DuckDuckGo API | Search query (topic) | [duckduckgo.com/privacy](https://duckduckgo.com/privacy) |
| RSS feeds (public) | HTTP GET only, no auth | Per individual publisher |
| Anthropic / OpenAI / Google / Mistral / Groq / OpenRouter | System prompt + research text | Per provider's API terms |
| SMTP provider (Gmail etc.) | Email content | Per your provider's terms |
| Slack (optional) | Report summary (300 chars max) | [slack.com/privacy](https://slack.com/privacy) |

## Data retention

- `genesis_runs.db` — retained indefinitely until manually deleted
- `report_*.html` / `report_*.json` — retained locally until manually deleted
- No data is sent to Saikou.tech servers

## Your rights (GDPR / CCPA)

To delete all stored run data:
```bash
rm genesis_runs.db report_*.html report_*.json
```

This pipeline is a self-hosted tool. All data stays on the machine where it runs.

## Contact

privacy@saikou.tech

