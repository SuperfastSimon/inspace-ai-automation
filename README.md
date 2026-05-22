# ⚡ Saikou.tech — AI Automation Hub

> **InSpace AI Automation Specialist Application** · Built by Martin Hatch, Founder @ [Saikou.tech](https://saikou.tech)

6 autonomous AI automation concepts demonstrating production-ready pipeline thinking.

## 🌐 Live Dashboard
**[Open Dashboard →](https://superfastsimon.github.io/inspace-ai-automation/)**

## Concepts

| # | Agent | Schedule | Key Tech |
|---|-------|----------|----------|
| 1 | [LLM Brand Entity Monitor](./concept1_llm_brand_monitor) | Daily 08:00 | Claude + GPT-4o parallel · Slack Block Kit |
| 2 | [Structured Data Auto-Injector](./concept2_structured_data_injector) | On-demand CLI | extruct · schema.org · WordPress REST |
| 3 | [Content Freshness Guardian](./concept3_content_freshness) | Weekly Mon 07:00 | Staleness scoring · Jinja2 HTML email |
| 4 | [Internal Linking Intelligence](./concept4_internal_linking) | On-demand CLI | OpenAI embeddings · cosine similarity · Google Sheets |
| 5 | [Competitor GEO Tracker](./concept5_competitor_geo) | Weekly Fri 06:00 | LLM visibility · SQLite trends · Markdown reports |
| 6 | [Genesis Pipeline](./genesis_pipeline) | Weekly Fri 06:00 + dispatch | Jina → 3× Sonnet → Opus → SMTP |

## Quick Start

```bash
git clone https://github.com/SuperfastSimon/inspace-ai-automation
cd inspace-ai-automation/<concept-folder>
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python main.py --run-now
```

## GitHub Actions

- **Push to main** → syntax check all agents
- **workflow_dispatch** → run Genesis Pipeline with custom topic + email
- **Weekly schedule** → automated Friday 06:00 Genesis report

Add secrets in repo Settings → Secrets: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `SMTP_USER`, `SMTP_PASS`
