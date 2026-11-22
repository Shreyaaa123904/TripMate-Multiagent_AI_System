"""Backend against a real PostgreSQL container (Docker)."""

import pytest

import backend
from tests import fakes
from tests.conftest import wait_until

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_backend(postgres_factory, monkeypatch):
    pg = postgres_factory()
    monkeypatch.setenv("DATABASE_URL", pg.url)
    monkeypatch.setattr(backend, "_travel_graph", None)
    monkeypatch.setattr(backend, "_pool", None)
    monkeypatch.setattr(backend, "_checkpointer", None)
    fakes_llm = fakes.FakeListChatModel(responses=[fakes.ITINERARY, fakes.FINAL_ANSWER])
    monkeypatch.setattr(backend, "get_llm", lambda: fakes_llm)
    monkeypatch.setattr(backend, "search_flights", lambda q: "Fake flight results")
    monkeypatch.setattr(backend, "tavily_search", lambda q: "Fake hotel results")
    backend.init_backend(force=True)
    yield pg
    backend.close_backend()


def _app_connections(pg):
    return pg.query(
        "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'tripmate'"
    )[0][0]


def test_pool_init_creates_checkpoint_tables(pg_backend):
    tables = {r[0] for r in pg_backend.query(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    assert {"checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"} <= tables
    assert backend.check_database() is True


def test_checkpoint_saved_and_restored_for_existing_thread(pg_backend):
    first = backend.run_travel_agent("Plan a 7 days Japan trip")
    tid = first["thread_id"]
    assert first["llm_calls"] == 2
    assert backend.thread_exists(tid)
    assert pg_backend.query("SELECT count(*) FROM checkpoints WHERE thread_id = %s", tid)[0][0] > 0

    state = backend.get_graph().get_state({"configurable": {"thread_id": tid}})
    first_count = len(state.values["messages"])

    second = backend.run_travel_agent("Make it 5 days", thread_id=tid)
    assert second["thread_id"] == tid
    state = backend.get_graph().get_state({"configurable": {"thread_id": tid}})
    contents = [m.content for m in state.values["messages"]]
    assert len(contents) > first_count
    assert "Plan a 7 days Japan trip" in contents and "Make it 5 days" in contents


def test_thread_survives_backend_restart(pg_backend):
    tid = backend.run_travel_agent("Plan a 7 days Japan trip")["thread_id"]
    backend.close_backend()
    backend.init_backend(force=True)
    assert backend.thread_exists(tid)
    assert not backend.thread_exists("user_" + "0" * 32)


def test_pool_reconnects_after_connections_killed(pg_backend):
    assert backend.check_database()
    pg_backend.query(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name = 'tripmate'")
    assert wait_until(backend.check_database, 10)
    assert backend.run_travel_agent("Plan a trip to Tokyo")["llm_calls"] == 2


def test_pool_closed_on_shutdown_leaves_no_connections(pg_backend):
    backend.run_travel_agent("Plan a trip to Tokyo")
    assert _app_connections(pg_backend) >= 1
    backend.close_backend()
    assert wait_until(lambda: _app_connections(pg_backend) == 0, 10)
