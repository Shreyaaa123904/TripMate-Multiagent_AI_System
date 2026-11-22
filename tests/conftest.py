import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set TRIPMATE_REQUIRE_INTEGRATION=1 (as CI does) to fail instead of skip
# when Docker or the browser isn't available.
REQUIRE_INTEGRATION = os.getenv("TRIPMATE_REQUIRE_INTEGRATION") == "1"
POSTGRES_IMAGE = os.getenv("TRIPMATE_TEST_POSTGRES_IMAGE", "postgres:16-alpine")


def skip_or_fail(reason: str):
    if REQUIRE_INTEGRATION:
        pytest.fail(reason)
    pytest.skip(reason)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(predicate, timeout: float, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


class PostgresContainer:
    """Disposable PostgreSQL container on a fixed localhost port."""

    password = "tripmate-test"
    database = "tripmate"

    def __init__(self):
        self.name = f"tripmate-test-pg-{uuid.uuid4().hex[:8]}"
        self.port = free_port()
        self.url = (
            f"postgresql://postgres:{self.password}@127.0.0.1:{self.port}/"
            f"{self.database}?sslmode=disable"
        )

    def _docker(self, *args, check=True):
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)

    def create(self):
        self._docker(
            "run", "-d", "--name", self.name,
            "-e", f"POSTGRES_PASSWORD={self.password}",
            "-e", f"POSTGRES_DB={self.database}",
            "-p", f"127.0.0.1:{self.port}:5432",
            POSTGRES_IMAGE,
        )
        self.wait_ready()

    def is_ready(self) -> bool:
        import psycopg
        with psycopg.connect(self.url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True

    def wait_ready(self, timeout: float = 60):
        if not wait_until(self.is_ready, timeout):
            raise RuntimeError("PostgreSQL test container did not become ready")

    def stop(self):
        self._docker("stop", "-t", "2", self.name)

    def start(self):
        self._docker("start", self.name)
        self.wait_ready()

    def remove(self):
        self._docker("rm", "-f", "-v", self.name, check=False)

    def query(self, sql: str, *params):
        import psycopg
        with psycopg.connect(self.url, autocommit=True) as conn:
            return conn.execute(sql, params).fetchall()


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture
def postgres_factory():
    if not _docker_available():
        skip_or_fail("Docker is not available for PostgreSQL integration tests")
    created = []

    def make():
        pg = PostgresContainer()
        created.append(pg)
        pg.create()
        return pg

    yield make
    for pg in created:
        pg.remove()


class AppServer:
    """Runs a uvicorn app in a subprocess (real server, real lifespan)."""

    def __init__(self, target: str, env: dict):
        self.target = target
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.env = {**os.environ, **env}
        self.proc = None
        self.log_path = ROOT / f".pytest-app-{self.port}.log"

    def start(self, wait_for_health: bool = True):
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", self.target, "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=ROOT, env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
        )
        if wait_for_health:
            import httpx
            ok = wait_until(lambda: httpx.get(self.base_url + "/health", timeout=2).status_code == 200, 30)
            if not ok:
                raise RuntimeError("App did not start:\n" + self.logs())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc:
            self._log.close()

    def logs(self) -> str:
        try:
            return self.log_path.read_text()
        except FileNotFoundError:
            return ""

    def cleanup(self):
        self.stop()
        self.log_path.unlink(missing_ok=True)
