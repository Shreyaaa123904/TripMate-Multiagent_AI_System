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
| `app.py` | FastAPI app: UI route, `/api/travel`, `/health` |
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
| `GET` | `/health` | – | `{"status": "ok", ...}` |

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

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run without API keys or a database. Known bugs are tracked as `xfail` tests until fixed. CI runs the suite on every pull request (`.github/workflows/ci.yml`).
