# TripMate AI

A multi-agent AI travel planner. A user describes a trip in natural language and a LangGraph pipeline gathers live flight data, hotel information and generates a day-by-day itinerary with a budget estimate.

## Architecture

```
Browser (templates/index.html, static/script.js)
        │  POST /api/travel {message, thread_id}
        ▼
FastAPI (app.py)
        │  run_travel_agent()
        ▼
LangGraph pipeline (backend.py)
  START → flight_agent → hotel_agent → itinerary_agent → final_agent → END
            │              │              │                 │
     AviationStack      Tavily        Groq LLM          Groq LLM
  (tools/flight_tool) (tools/tavily_tool)
        │
PostgreSQL — LangGraph PostgresSaver checkpoints per thread_id
```

| Path | Purpose |
|---|---|
| `app.py` | FastAPI app: UI route, `/api/travel`, `/health`, `/health/ready`, validation, rate limit, security headers |
| `config.py` | Settings read from the environment |
| `security.py` | Thread-ID helpers and the in-memory rate limiter |
| `backend.py` | LangGraph graph, Groq LLM, PostgresSaver checkpointer |
| `tools/flight_tool.py` | Natural-language route → IATA codes → AviationStack live flights |
| `tools/tavily_tool.py` | Tavily web search for hotels |
| `templates/`, `static/` | HTML/CSS/JS frontend |
| `test.py` | Interactive command-line run of the agent |
| `tests/` | Automated pytest suite |

### API

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/` | – | Web UI |
| `POST` | `/api/travel` | `{"message": str, "thread_id": str \| null}` | `{success, thread_id, answer, flight_results, hotel_results, itinerary, llm_calls}` |
| `GET` | `/health` | – | Liveness. Always `200` while the process runs: `{"status": "ok" \| "degraded", "message", "checks": {"database": "ok" \| "unavailable"}}` |
| `GET` | `/health/ready` | – | Readiness. `200` when the database is reachable, otherwise `503` |

Errors keep the `{"success": false, "error": "..."}` shape: `400` empty message, `413` body too large, `422` invalid input (message over 2,000 characters, bad `thread_id`, wrong types), `429` rate limited (with `Retry-After`), `503` database unavailable, `504` upstream timeout, `500` anything else. Server errors also include a `request_id` (also sent as the `X-Request-ID` header) that matches the server log entry; internal details are never returned.

## Requirements

- Python 3.11+
- PostgreSQL (local or hosted, e.g. Render)
- API keys: [Groq](https://console.groq.com), [Tavily](https://tavily.com), [AviationStack](https://aviationstack.com)

## Configuration

Copy the example file and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq LLM API key |
| `TAVILY_API_KEY` | yes | Tavily search API key |
| `AVIATIONSTACK_API_KEY` | yes | AviationStack API key |
| `DATABASE_URL` | yes | PostgreSQL connection string (`sslmode=require` is added automatically) |
| `DEFAULT_ORIGIN_IATA` | no | Origin airport when a query names only a destination (default `DAC`) |
| `DB_POOL_MIN` / `DB_POOL_MAX` | no | Connection pool size per process (default `1` / `5`) |
| `DB_SSLMODE` | no | `sslmode` added when the URL has none (default `require`) |
| `RATE_LIMIT_PER_MINUTE` | no | `/api/travel` requests per client IP per minute (default `10`, `0` disables) |
| `MAX_MESSAGE_CHARS` | no | Maximum message length (default `2000`) |
| `FORWARDED_ALLOW_IPS` | no | Proxy IPs trusted to set `X-Forwarded-For` (default `127.0.0.1`) |

All optional settings are listed in `.env.example`.

`.env` is git-ignored and excluded from Docker builds. Never commit real keys.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # http://127.0.0.1:8000
```

## Run with Docker

```bash
docker build -t tripmate-ai .
docker run --env-file .env -p 8000:8000 tripmate-ai
```

The container runs as a non-root user and exposes a health check on `/health`.

## Deployment notes

**Database.** The app starts even if PostgreSQL is down: the UI and `/health` work, `/api/travel` returns `503`, and the app reconnects automatically in the background. Use `/health/ready` as the platform health-check path (e.g. Render's *Health Check Path*). Each process opens at most `DB_POOL_MAX` connections (default 5); keep `DB_POOL_MAX × WEB_CONCURRENCY × instances` below your database's connection limit. The app logs a warning at startup if it could exceed `max_connections`. The pool is closed on shutdown.

**Client IPs and rate limiting.** The rate limit is keyed by client IP. Behind a reverse proxy (Render, Nginx), set `FORWARDED_ALLOW_IPS` to the proxy's address(es) so uvicorn uses the `X-Forwarded-For` value appended by that proxy. Headers from any other address are ignored, so clients can't spoof their IP. If a forwarded header arrives from an untrusted address, the app logs that peer address once, which shows the value to put in `FORWARDED_ALLOW_IPS`. If it isn't set, all users behind the proxy share one limit. The limiter is in memory and per process, so with several processes or instances the effective limit is multiplied, and it resets on restart.

**Thread IDs are secrets.** A `thread_id` is a 128-bit random value generated by the server and stored in the browser's `localStorage`. Anyone who has a valid `thread_id` can continue that conversation, and earlier trip details could be echoed back to them. IDs are never put in URLs, listed by any endpoint, or written in full to logs (only a short hash). Unknown IDs are replaced with a new one. Real ownership checks need user accounts and are out of scope until authentication is added.

**Frontend security.** Model output is rendered as Markdown and sanitised with DOMPurify (vendored in `static/vendor/`, with `marked`); if either fails to load, the answer is shown as plain text. A Content-Security-Policy and other security headers are sent on every page.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

No API keys are needed: the LLM and search providers are faked.

| Suite | Files | Needs |
|---|---|---|
| Unit and API | `tests/test_*.py` | nothing |
| PostgreSQL integration | `tests/test_db_integration.py` | Docker |
| DB outage / recovery / restart (real server) | `tests/e2e/` | Docker |
| Browser (Markdown, XSS payloads, CSP, Copy, PDF) | `tests/browser/` | `python -m playwright install chromium`, internet (html2pdf CDN) |

Docker and browser tests are skipped locally when unavailable; CI sets `TRIPMATE_REQUIRE_INTEGRATION=1` so they fail instead. CI runs every suite on each pull request (`.github/workflows/ci.yml`).
