import asyncio
import logging
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import app as app_module
import backend


FAKE_RESULT = {
    "answer": "## Trip Summary\nHave fun.",
    "flight_results": "flights",
    "hotel_results": "hotels",
    "itinerary": "itinerary",
    "llm_calls": 2,
}


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    app_module.rate_limiter.reset()
    monkeypatch.setattr(app_module.rate_limiter, "limit", 10)
    app_module._db_health.update(ok=False, checked_at=0.0)
    monkeypatch.setattr(backend, "thread_exists", lambda tid: tid.startswith("user_known"))
    yield


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_run(user_input, thread_id=None):
        seen.append((user_input, thread_id))
        return {"thread_id": thread_id or "user_" + "a" * 32, **FAKE_RESULT}

    monkeypatch.setattr(backend, "run_travel_agent", fake_run)
    return seen


@pytest.fixture
def client():
    return TestClient(app_module.app)


def post(client, **body):
    return client.post("/api/travel", json=body)


# --- Response compatibility -------------------------------------------------

def test_success_response_shape_unchanged(client, calls):
    r = post(client, message="Plan a 7 days Japan trip")
    assert r.status_code == 200
    assert r.json() == {"success": True, "thread_id": "user_" + "a" * 32, **FAKE_RESULT}
    assert calls == [("Plan a 7 days Japan trip", None)]


def test_message_is_stripped(client, calls):
    post(client, message="  hello  ")
    assert calls[0][0] == "hello"


# --- Validation ---------------------------------------------------------------

@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_message_returns_400_as_before(client, calls, message):
    r = post(client, message=message)
    assert r.status_code == 400
    assert r.json()["success"] is False
    assert r.json()["error"] == "Message cannot be empty."
    assert calls == []


def test_message_at_limit_accepted(client, calls):
    r = post(client, message="a" * app_module.config.MAX_MESSAGE_CHARS)
    assert r.status_code == 200


def test_message_over_limit_rejected(client, calls):
    r = post(client, message="a" * (app_module.config.MAX_MESSAGE_CHARS + 1))
    assert r.status_code == 422
    assert r.json()["success"] is False
    assert "too long" in r.json()["error"]
    assert calls == []


@pytest.mark.parametrize("body", [{}, {"message": 123}, {"message": None}, {"message": ["x"]}])
def test_wrong_types_return_json_422(client, calls, body):
    r = client.post("/api/travel", json=body)
    assert r.status_code == 422
    assert r.json()["success"] is False
    assert isinstance(r.json()["error"], str)


def test_oversized_body_rejected(client, calls):
    r = client.post(
        "/api/travel",
        content=b'{"message": "' + b"a" * 20000 + b'"}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert calls == []


# --- Thread ids ----------------------------------------------------------------

@pytest.mark.parametrize(
    "thread_id",
    ["../etc/passwd", "a b", "x" * 65, "user_<script>", "drop;table", "ü"],
)
def test_invalid_thread_ids_rejected(client, calls, thread_id):
    r = post(client, message="hi", thread_id=thread_id)
    assert r.status_code == 422
    assert r.json()["error"] == "Invalid thread_id."
    assert calls == []


@pytest.mark.parametrize("thread_id", [None, ""])
def test_missing_thread_id_starts_new_thread(client, calls, thread_id):
    r = post(client, message="hi", thread_id=thread_id)
    assert r.status_code == 200
    assert calls[0][1] is None


def test_existing_thread_id_is_reused(client, calls):
    tid = "user_known" + "b" * 22
    r = post(client, message="hi", thread_id=tid)
    assert r.status_code == 200
    assert calls[0][1] == tid
    assert r.json()["thread_id"] == tid


def test_unknown_thread_id_replaced_with_new_one(client, calls):
    r = post(client, message="hi", thread_id="test_user")
    assert r.status_code == 200
    assert calls[0][1] is None
    assert r.json()["thread_id"] != "test_user"


# --- Rate limiting -------------------------------------------------------------

def test_rate_limit_returns_429_with_retry_after(client, calls):
    for _ in range(10):
        assert post(client, message="hi").status_code == 200
    r = post(client, message="hi")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert r.json()["success"] is False
    assert len(calls) == 10


def test_health_not_rate_limited(client, calls):
    for _ in range(11):
        post(client, message="hi")
    assert client.get("/health").status_code == 200


# --- Safe errors ---------------------------------------------------------------

def test_unexpected_error_is_generic_and_logged(client, monkeypatch, caplog):
    secret = "postgresql://admin:hunter2@db.internal/tripmate"

    def boom(**kwargs):
        raise RuntimeError(f"connection failed for {secret}")

    monkeypatch.setattr(backend, "run_travel_agent", boom)
    with caplog.at_level(logging.ERROR, logger="tripmate.app"):
        r = post(client, message="hi")

    assert r.status_code == 500
    body = r.json()
    assert body["success"] is False
    assert body["error"] == app_module.GENERIC_ERROR_MESSAGE
    assert "hunter2" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert body["request_id"] in caplog.text
    assert "Traceback" in caplog.text


def test_database_unavailable_returns_503(client, monkeypatch):
    def down(**kwargs):
        raise backend.BackendUnavailable("Database is unavailable.")

    monkeypatch.setattr(backend, "run_travel_agent", down)
    r = post(client, message="hi")
    assert r.status_code == 503
    assert r.json()["error"] == app_module.UNAVAILABLE_MESSAGE


def test_llm_timeout_returns_504(client, monkeypatch):
    def slow(**kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(backend, "run_travel_agent", slow)
    r = post(client, message="hi")
    assert r.status_code == 504
    assert r.json()["error"] == app_module.TIMEOUT_MESSAGE


# --- Security headers / template -----------------------------------------------

def test_security_headers_on_page_and_api(client, calls):
    for r in (client.get("/"), post(client, message="hi"), client.get("/health")):
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "same-origin"
        csp = r.headers["Content-Security-Policy"]
        assert "script-src 'self' https://cdnjs.cloudflare.com" in csp
        assert "'unsafe-eval'" not in csp
        assert "object-src 'none'" in csp
        assert r.headers["X-Request-ID"]


def test_docs_still_available_without_csp(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "Content-Security-Policy" not in r.headers


def test_template_has_no_inline_handlers_and_uses_vendored_scripts(client):
    html = client.get("/").text
    assert "onclick=" not in html
    assert "/static/vendor/marked.min.js" in html
    assert "/static/vendor/purify.min.js" in html
    assert "cdn.jsdelivr.net" not in html
    assert 'integrity="sha512-' in html


@pytest.mark.parametrize("path", ["/static/vendor/marked.min.js", "/static/vendor/purify.min.js"])
def test_vendored_scripts_served(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_favicon_no_content(client):
    assert client.get("/favicon.ico").status_code == 204


# --- Health ------------------------------------------------------------------------

def test_health_degraded_when_backend_not_ready(client, monkeypatch):
    monkeypatch.setattr(backend, "is_backend_ready", lambda: False)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "degraded",
        "message": "AI Travel Planner API is running",
        "checks": {"database": "unavailable"},
    }


def test_health_ok_and_cached(client, monkeypatch):
    probes = []
    monkeypatch.setattr(backend, "is_backend_ready", lambda: True)
    monkeypatch.setattr(backend, "check_database", lambda timeout=2.0: probes.append(1) or True)
    for _ in range(5):
        r = client.get("/health")
        assert r.json()["status"] == "ok"
        assert r.json()["checks"]["database"] == "ok"
    assert len(probes) == 1


def test_ready_200_when_database_ok(client, monkeypatch):
    monkeypatch.setattr(backend, "is_backend_ready", lambda: True)
    monkeypatch.setattr(backend, "check_database", lambda timeout=2.0: True)
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_ready_503_when_database_down(client, monkeypatch):
    monkeypatch.setattr(backend, "is_backend_ready", lambda: False)

    def fail(force=False):
        raise backend.BackendUnavailable("Database is unavailable.")

    monkeypatch.setattr(backend, "init_backend", fail)
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "checks": {"database": "unavailable"}}


def test_health_output_has_no_connection_details(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://admin:hunter2@db.internal/tripmate")
    monkeypatch.setattr(backend, "is_backend_ready", lambda: False)
    for path in ("/health", "/health/ready"):
        text = client.get(path).text
        assert "hunter2" not in text and "db.internal" not in text


# --- Non-blocking execution ------------------------------------------------------

def test_concurrent_requests_not_blocked(monkeypatch):
    def slow_run(user_input, thread_id=None):
        time.sleep(1)
        return {"thread_id": "user_" + "c" * 32, **FAKE_RESULT}

    monkeypatch.setattr(backend, "run_travel_agent", slow_run)
    monkeypatch.setattr(backend, "is_backend_ready", lambda: False)

    async def scenario():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            start = time.perf_counter()
            health_elapsed = {}

            async def health():
                await asyncio.sleep(0.2)
                t0 = time.perf_counter()
                r = await ac.get("/health")
                health_elapsed["t"] = time.perf_counter() - t0
                return r

            results = await asyncio.gather(
                ac.post("/api/travel", json={"message": "a"}),
                ac.post("/api/travel", json={"message": "b"}),
                health(),
            )
            return results, time.perf_counter() - start, health_elapsed["t"]

    results, elapsed, health_elapsed = asyncio.run(scenario())
    assert [r.status_code for r in results] == [200, 200, 200]
    assert elapsed < 1.8, f"requests ran serially ({elapsed:.2f}s)"
    assert health_elapsed < 0.5, f"/health blocked for {health_elapsed:.2f}s"
