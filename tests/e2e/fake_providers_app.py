"""Test-only entrypoint: the real app with fake LLM/search providers."""

import backend
from tests import fakes

fakes.install(backend)

from app import app  # noqa: E402,F401
