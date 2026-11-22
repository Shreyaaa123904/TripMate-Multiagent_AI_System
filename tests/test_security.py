import re

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import security
from security import RateLimiter, is_valid_thread_id, new_thread_id, thread_ref


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_limiter_allows_up_to_limit_then_blocks():
    clock = FakeClock()
    rl = RateLimiter(3, 60, clock=clock)
    assert [rl.hit("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = rl.hit("a")
    assert not allowed and 1 <= retry <= 60


def test_limiter_keys_are_independent():
    rl = RateLimiter(1, 60, clock=FakeClock())
    assert rl.hit("a")[0]
    assert not rl.hit("a")[0]
    assert rl.hit("b")[0]


def test_limiter_window_slides():
    clock = FakeClock()
    rl = RateLimiter(2, 60, clock=clock)
    rl.hit("a")
    clock.now += 30
    rl.hit("a")
    assert not rl.hit("a")[0]
    clock.now += 31  # first hit expired
    assert rl.hit("a")[0]


def test_limiter_retry_after_is_accurate():
    clock = FakeClock()
    rl = RateLimiter(1, 60, clock=clock)
    rl.hit("a")
    clock.now += 45
    assert rl.hit("a") == (False, 15)


def test_limiter_disabled_with_zero_limit():
    rl = RateLimiter(0, 60, clock=FakeClock())
    assert all(rl.hit("a")[0] for _ in range(100))


def test_thread_ids():
    tid = new_thread_id()
    assert re.fullmatch(r"user_[0-9a-f]{32}", tid)
    assert len({new_thread_id() for _ in range(1000)}) == 1000
    assert is_valid_thread_id(tid)
    assert is_valid_thread_id("test_user")
    for bad in ["", "a b", "x" * 65, "../x", "a\nb"]:
        assert not is_valid_thread_id(bad)


def test_thread_ref_does_not_reveal_id():
    tid = new_thread_id()
    ref = thread_ref(tid)
    assert len(ref) == 12 and ref not in tid
    assert thread_ref(None) == "-"


# --- Client IP resolution behind a proxy (uvicorn ProxyHeadersMiddleware) -------

def _proxied_app(trusted: str):
    inner = FastAPI()

    @inner.get("/ip")
    async def ip(request: Request):
        return {"ip": request.client.host}

    return ProxyHeadersMiddleware(inner, trusted_hosts=trusted)


def _ip(trusted, peer, xff=None):
    client = TestClient(_proxied_app(trusted), client=(peer, 50000))
    headers = {"X-Forwarded-For": xff} if xff else {}
    return client.get("/ip", headers=headers).json()["ip"]


def test_untrusted_peer_cannot_spoof_forwarded_for():
    assert _ip("127.0.0.1", "203.0.113.9", "1.2.3.4") == "203.0.113.9"


def test_trusted_proxy_forwarded_for_is_used():
    assert _ip("10.0.0.0/8", "10.1.2.3", "198.51.100.7") == "198.51.100.7"


def test_client_prepended_fake_entries_are_ignored():
    # Attacker sends "X-Forwarded-For: 1.2.3.4"; the proxy appends the real IP.
    assert _ip("10.0.0.0/8", "10.1.2.3", "1.2.3.4, 198.51.100.7") == "198.51.100.7"


def test_chain_of_trusted_proxies_resolves_rightmost_untrusted():
    assert _ip("10.0.0.0/8", "10.1.2.3", "1.2.3.4, 198.51.100.7, 10.9.9.9") == "198.51.100.7"


def test_spoofer_cannot_escape_rate_limit_bucket():
    rl = RateLimiter(2, 60, clock=FakeClock())
    keys = [_ip("10.0.0.0/8", "10.1.2.3", f"{n}.{n}.{n}.{n}, 198.51.100.7") for n in range(1, 5)]
    assert set(keys) == {"198.51.100.7"}
    assert [rl.hit(k)[0] for k in keys] == [True, True, False, False]


def test_default_trusts_nothing_external():
    assert security  # module import sanity
    assert _ip("127.0.0.1", "10.1.2.3", "198.51.100.7") == "10.1.2.3"
