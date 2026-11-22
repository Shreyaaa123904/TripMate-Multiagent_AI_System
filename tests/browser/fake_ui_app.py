"""Test-only entrypoint: real app/UI, agent replaced by a fixed Markdown answer."""

from pathlib import Path

import backend

ANSWER = (Path(__file__).parent / "fixture_answer.md").read_text()

backend.init_backend = lambda force=False: None
backend.is_backend_ready = lambda: True
backend.check_database = lambda timeout=2.0: True
backend.thread_exists = lambda thread_id: True
backend.run_travel_agent = lambda user_input, thread_id=None: {
    "thread_id": thread_id or "user_" + "f" * 32,
    "answer": ANSWER,
    "flight_results": "",
    "hotel_results": "",
    "itinerary": "",
    "llm_calls": 2,
}

from app import app  # noqa: E402,F401
