import logging

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import backend


@pytest.fixture(autouse=True)
def clean_backend(monkeypatch):
    monkeypatch.setattr(backend, "_travel_graph", None)
    monkeypatch.setattr(backend, "_pool", None)
    monkeypatch.setattr(backend, "_checkpointer", None)
    monkeypatch.setattr(backend, "_last_init_attempt", 0.0)
    yield


@pytest.mark.parametrize(
    "url, sslmode, expected",
    [
        ("postgresql://u:p@h/db", "require", "postgresql://u:p@h/db?sslmode=require"),
        ("postgresql://u:p@h/db?x=1", "require", "postgresql://u:p@h/db?x=1&sslmode=require"),
        ("postgresql://u:p@h/db?sslmode=disable", "require", "postgresql://u:p@h/db?sslmode=disable"),
        ("postgresql://u:p@h/db", "disable", "postgresql://u:p@h/db?sslmode=disable"),
    ],
)
def test_database_url_sslmode(monkeypatch, url, sslmode, expected):
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(backend.config, "DB_SSLMODE", sslmode)
    assert backend.get_database_url() == expected


def test_missing_database_url_is_backend_unavailable(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(backend.BackendUnavailable):
        backend.init_backend(force=True)
    assert not backend.is_backend_ready()


def test_missing_groq_key_is_clear_error(monkeypatch):
    monkeypatch.setattr(backend, "_llm", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(backend.BackendUnavailable, match="GROQ_API_KEY"):
        backend.get_llm()


def test_import_does_not_connect_or_require_secrets():
    # Importing backend (done above) must not open a connection.
    assert backend._pool is None


class FakePool:
    instances = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.close_timeout = None
        FakePool.instances.append(self)

    def open(self, wait=False, timeout=None):
        if self.kwargs.get("fail"):
            raise RuntimeError("could not connect to server: password=hunter2")

    def close(self, timeout=None):
        self.closed = True
        self.close_timeout = timeout

    check_connection = staticmethod(lambda conn: None)


class FakeSaver(InMemorySaver):
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def setup(self):
        pass


@pytest.fixture
def fake_pool(monkeypatch):
    FakePool.instances = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setattr(backend, "ConnectionPool", FakePool)
    monkeypatch.setattr(backend, "PostgresSaver", FakeSaver)
    monkeypatch.setattr(backend, "_check_connection_budget", lambda pool: None)
    return FakePool


def test_init_opens_pool_with_safe_settings_and_close_closes_it(fake_pool):
    backend.init_backend(force=True)
    assert backend.is_backend_ready()
    pool = fake_pool.instances[0]
    kw = pool.kwargs
    assert kw["min_size"] == backend.config.DB_POOL_MIN
    assert kw["max_size"] == backend.config.DB_POOL_MAX
    assert kw["kwargs"]["autocommit"] is True
    assert kw["kwargs"]["prepare_threshold"] == 0
    assert kw["kwargs"]["application_name"] == "tripmate"
    assert kw["check"] is not None

    backend.close_backend()
    assert pool.closed and pool.close_timeout == 5
    assert not backend.is_backend_ready()


def test_init_failure_closes_pool_and_hides_details(fake_pool, monkeypatch, caplog):
    orig_init = FakePool.__init__

    def failing_init(self, *a, **kw):
        orig_init(self, *a, **{**kw, "fail": True})

    monkeypatch.setattr(FakePool, "__init__", failing_init)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(backend.BackendUnavailable) as exc:
            backend.init_backend(force=True)
    assert "hunter2" not in str(exc.value)
    assert "hunter2" not in caplog.text
    assert fake_pool.instances[0].closed
    assert not backend.is_backend_ready()


def test_init_retries_are_throttled(fake_pool, monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(backend.BackendUnavailable):
        backend.init_backend()
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    with pytest.raises(backend.BackendUnavailable, match="retried"):
        backend.init_backend()
    backend.init_backend(force=True)
    assert backend.is_backend_ready()


def test_get_graph_initialises_lazily(fake_pool):
    assert backend.get_graph() is not None
    assert len(fake_pool.instances) == 1
    backend.get_graph()
    assert len(fake_pool.instances) == 1
    backend.close_backend()


def test_check_database_false_without_pool():
    assert backend.check_database() is False


class _Conn:
    def __init__(self, max_conn):
        self.max_conn = max_conn

    def execute(self, sql):
        return self

    def fetchone(self):
        return {"max_connections": str(self.max_conn)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _BudgetPool:
    def __init__(self, max_conn):
        self.max_conn = max_conn

    def connection(self):
        return _Conn(self.max_conn)


def test_connection_budget_warning(monkeypatch, caplog):
    monkeypatch.setattr(backend.config, "DB_POOL_MAX", 5)
    monkeypatch.setattr(backend.config, "WEB_CONCURRENCY", 4)
    with caplog.at_level(logging.WARNING):
        backend._check_connection_budget(_BudgetPool(25))
    assert "exceed" in caplog.text


def test_connection_budget_ok(monkeypatch, caplog):
    monkeypatch.setattr(backend.config, "DB_POOL_MAX", 5)
    monkeypatch.setattr(backend.config, "WEB_CONCURRENCY", 1)
    with caplog.at_level(logging.WARNING):
        backend._check_connection_budget(_BudgetPool(100))
    assert "exceed" not in caplog.text
