"""Central configuration, read once from the environment (.env supported)."""

import logging
import os

import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        logging.getLogger("tripmate.config").warning(
            "Ignoring invalid integer for %s; using %s", name, default
        )
        return default


DB_SSLMODE = os.getenv("DB_SSLMODE", "require")
DB_POOL_MIN = _int("DB_POOL_MIN", 1)
DB_POOL_MAX = _int("DB_POOL_MAX", 5)
DB_POOL_TIMEOUT_SECONDS = _int("DB_POOL_TIMEOUT_SECONDS", 10)
DB_CONNECT_TIMEOUT_SECONDS = _int("DB_CONNECT_TIMEOUT_SECONDS", 5)
DB_INIT_RETRY_SECONDS = _int("DB_INIT_RETRY_SECONDS", 5)
WEB_CONCURRENCY = _int("WEB_CONCURRENCY", 1)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 2)

MAX_MESSAGE_CHARS = _int("MAX_MESSAGE_CHARS", 2000)
MAX_REQUEST_BYTES = _int("MAX_REQUEST_BYTES", 16 * 1024)
RATE_LIMIT_PER_MINUTE = _int("RATE_LIMIT_PER_MINUTE", 10)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(LOG_LEVEL)
        return
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
