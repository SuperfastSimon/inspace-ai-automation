# Genesis Pipeline — AI Research & Content Intelligence

Self-sufficient, single-file autonomous pipeline that:
1. Researches any topic via Jina web search
2. Runs 3 parallel LLM analyses (Executive Summary / Opportunities / Action Plan)
3. Combines results
4. Formats a polished HTML email report via Claude Opus
5. Delivers via SMTP (or saves locally if SMTP not configured)

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys

# Run once (saves HTML report locally — no SMTP needed):
python genesis_pipeline.py --topic "Saikou.tech AI automation" --email you@example.com --dry-run

# Run once and send email:
python genesis_pipeline.py --topic "n8n workflow automation trends" --email you@example.com

# Run on weekly schedule (every Friday 06:00 Amsterdam time):
python genesis_pipeline.py --topic "AI content automation" --email you@example.com --schedule
```

## Requirements

- Python 3.11+
- At least `ANTHROPIC_API_KEY` set in `.env`
- `OPENAI_API_KEY` optional (used as fallback for opportunities analysis)
- SMTP credentials optional (without them, reports are saved as HTML files)

## Pipeline Architecture

```
Topic Input
    │
    ▼
Web Research (Jina Reader API)
    │
    ├──────────────────────────────────┐──────────────────────────────────┐
    ▼                                  ▼                                  ▼
Executive Summary              Top 5 Opportunities             90-Day Action Plan
(claude-sonnet-4-6)            (claude-sonnet-4-6)             (claude-sonnet-4-6)
    │                                  │                                  │
    └──────────────────────────────────┴──────────────────────────────────┘
                                       │
                                  Combine Results
                                       │
                                       ▼
                           Format HTML Report
                             (claude-opus-4-7)
                                       │
                                       ▼
                               Send via SMTP
                         (or save locally if no SMTP)
```
