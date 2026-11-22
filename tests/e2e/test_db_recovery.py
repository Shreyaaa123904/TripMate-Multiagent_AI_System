"""Real server + real PostgreSQL: outage at startup, recovery, persistence."""

import httpx
import pytest

from tests.conftest import AppServer, wait_until

pytestmark = pytest.mark.integration


def _env(pg):
    return {
        "DATABASE_URL": pg.url,
        "DB_INIT_RETRY_SECONDS": "1",
        "DB_POOL_TIMEOUT_SECONDS": "3",
        "DB_CONNECT_TIMEOUT_SECONDS": "2",
        "RATE_LIMIT_PER_MINUTE": "100",
        "GROQ_API_KEY": "",
    }


def test_startup_without_db_then_recovery_and_persistence(postgres_factory):
    pg = postgres_factory()
    pg.stop()

    server = AppServer("tests.e2e.fake_providers_app:app", _env(pg))
    try:
        # 1. Database down: app still starts and serves UI + liveness.
        server.start()
        base = server.base_url
        assert httpx.get(base + "/").status_code == 200
        health = httpx.get(base + "/health").json()
        assert health["status"] == "degraded"
        assert httpx.get(base + "/health/ready").status_code == 503
        r = httpx.post(base + "/api/travel", json={"message": "Plan a 7 days Japan trip"}, timeout=30)
        assert r.status_code == 503
        assert r.json()["success"] is False
        assert "database" in r.json()["error"].lower()
        assert pg.password not in r.text and "Traceback" not in r.text

        # 2. Database comes back: recovers without restarting the app.
        pg.start()
        assert wait_until(lambda: httpx.get(base + "/health/ready").status_code == 200, 20), server.logs()

        # 3. A plan succeeds with the same response shape.
        r = httpx.post(base + "/api/travel", json={"message": "Plan a 7 days Japan trip"}, timeout=30)
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"success", "thread_id", "answer", "flight_results", "hotel_results",
                             "itinerary", "llm_calls"}
        assert body["llm_calls"] == 2
        tid = body["thread_id"]

        # 4. Checkpoints are saved for the thread.
        assert pg.query("SELECT count(*) FROM checkpoints WHERE thread_id = %s", tid)[0][0] > 0
        before = pg.query("SELECT count(*) FROM checkpoints WHERE thread_id = %s", tid)[0][0]

        # 5. Same thread is restored and reused.
        r = httpx.post(base + "/api/travel", json={"message": "Make it 5 days", "thread_id": tid}, timeout=30)
        assert r.status_code == 200 and r.json()["thread_id"] == tid
        assert pg.query("SELECT count(*) FROM checkpoints WHERE thread_id = %s", tid)[0][0] > before

        # 6. Thread survives an application restart.
        server.stop()
        server.start()
        assert wait_until(lambda: httpx.get(base + "/health/ready").status_code == 200, 20)
        r = httpx.post(base + "/api/travel", json={"message": "Add Osaka", "thread_id": tid}, timeout=30)
        assert r.status_code == 200 and r.json()["thread_id"] == tid

        # 7. Mid-run outage → safe 503, then automatic reconnect.
        pg.stop()
        r = httpx.post(base + "/api/travel", json={"message": "Plan a trip to Tokyo"}, timeout=30)
        assert r.status_code == 503
        assert pg.password not in r.text
        assert httpx.get(base + "/health/ready").status_code == 503
        pg.start()
        assert wait_until(lambda: httpx.get(base + "/health/ready").status_code == 200, 20), server.logs()
        r = httpx.post(base + "/api/travel", json={"message": "Add Osaka", "thread_id": tid}, timeout=30)
        assert r.status_code == 200 and r.json()["thread_id"] == tid

        # 8. Graceful shutdown closes the pool.
        server.stop()
        assert wait_until(lambda: pg.query(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'tripmate'")[0][0] == 0, 10)
        assert "Database pool closed" in server.logs()
    finally:
        server.cleanup()
