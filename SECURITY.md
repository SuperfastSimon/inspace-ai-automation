# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (main branch) | ✅ |
| older commits | ❌ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email: security@saikou.tech  
Response time: within 48 hours  
Resolution target: within 7 days for critical issues

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

## Security Design

### API Key Handling
- All API keys are loaded exclusively from environment variables via `.env` (never hardcoded)
- `.env` is listed in `.gitignore` — never committed
- GitHub Actions secrets are used in CI (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.)
- Keys are never logged, printed, or included in reports

### Data Flow
- Research content (web scraped text) is passed to LLM APIs over HTTPS
- Email addresses are used only for SMTP delivery — never stored in logs or reports
- SQLite run log (`genesis_runs.db`) stores: topic, provider name, cost estimate, timestamps, and report hash — **no API keys, no email content, no PII**

### Email Delivery
- SMTP credentials loaded from env only
- TLS enforced (`starttls()`) for all SMTP connections
- No email content is stored locally (dry-run saves HTML report only)

### Network Requests
- All outbound requests use HTTPS
- Timeouts enforced on all HTTP calls (10–30s)
- No inbound ports opened — pipeline is outbound-only

### Dependencies
- Pinned minimum versions in `requirements.txt`
- Run `pip audit` or `safety check` regularly to scan for CVEs

## Known Limitations
- Token cost estimates are approximations (~±30%) — not suitable for billing without real usage objects from provider APIs
- The DuckDuckGo fallback uses an undocumented API endpoint — may break without notice
- RSS feeds are public sources — content is not verified for accuracy

