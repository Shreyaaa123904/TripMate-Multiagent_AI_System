import asyncio
import ipaddress
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

import config
import backend
from security import THREAD_ID_PATTERN, RateLimiter, thread_ref

config.configure_logging()
logger = logging.getLogger("tripmate.app")

BASE_DIR = Path(__file__).resolve().parent

UNAVAILABLE_MESSAGE = (
    "TripMate is temporarily unavailable because the database cannot be reached. "
    "Please try again in a moment."
)
TIMEOUT_MESSAGE = "The AI planner took too long to respond. Please try again."
GENERIC_ERROR_MESSAGE = "Something went wrong while planning your trip. Please try again."

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://cdnjs.cloudflare.com",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])
# FastAPI's interactive docs load Swagger/ReDoc from a CDN with inline scripts.
CSP_EXEMPT_PATHS = ("/docs", "/redoc")

rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)

_db_health = {"ok": False, "checked_at": 0.0}
DB_HEALTH_CACHE_SECONDS = 10
_proxy_warning_logged = False


async def _recover_backend(stop: asyncio.Event) -> None:
    """Keep retrying database initialisation until it succeeds."""
    while not stop.is_set() and not backend.is_backend_ready():
        try:
            await run_in_threadpool(backend.init_backend, True)
            logger.info("Database connection established")
            return
        except backend.BackendUnavailable as exc:
            logger.warning("Backend not ready (%s); retrying in %ss", exc, config.DB_INIT_RETRY_SECONDS)
        except Exception:
            logger.exception("Unexpected error while initialising backend")
        try:
            await asyncio.wait_for(stop.wait(), timeout=config.DB_INIT_RETRY_SECONDS)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = asyncio.Event()
    recovery = asyncio.create_task(_recover_backend(stop))
    try:
        yield
    finally:
        stop.set()
        recovery.cancel()
        try:
            await recovery
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Backend recovery task failed")
        await run_in_threadpool(backend.close_backend)


app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.1.0",
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


def _error(status_code: int, message: str, request: Request | None = None, headers=None):
    content = {"success": False, "error": message}
    if request is not None and getattr(request.state, "request_id", None):
        content["request_id"] = request.state.request_id
    return JSONResponse(status_code=status_code, content=content, headers=headers)


def _warn_untrusted_proxy(request: Request) -> None:
    """Log once if a forwarded header arrives from a peer we don't trust."""
    global _proxy_warning_logged
    if _proxy_warning_logged:
        return
    forwarded = request.headers.get("x-forwarded-for")
    client = request.client.host if request.client else None
    if not forwarded or not client:
        return
    entries = [h.strip() for h in forwarded.split(",")]
    if client in entries:
        return  # uvicorn trusted the proxy and resolved the client address
    try:
        is_private = ipaddress.ip_address(client).is_private
    except ValueError:
        is_private = False
    if is_private:
        _proxy_warning_logged = True
        logger.warning(
            "X-Forwarded-For received from untrusted proxy peer %s; all clients share its "
            "rate-limit bucket. If this is your platform's load balancer, add it to "
            "FORWARDED_ALLOW_IPS.",
            client,
        )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id

    if request.method == "POST" and request.url.path.startswith("/api/"):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > config.MAX_REQUEST_BYTES:
            response = _error(413, "Request is too large.", request)
            return _with_headers(response, request)
        _warn_untrusted_proxy(request)

    response = await call_next(request)
    return _with_headers(response, request)


def _with_headers(response: Response, request: Request) -> Response:
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if not request.url.path.startswith(CSP_EXEMPT_PATHS):
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/api/"):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    message = "Invalid request."
    for err in exc.errors():
        field = err.get("loc", [None])[-1]
        if field == "message" and err.get("type") == "string_too_long":
            message = f"Message is too long (maximum {config.MAX_MESSAGE_CHARS} characters)."
        elif field == "thread_id":
            message = "Invalid thread_id."
        elif field in ("message", "thread_id"):
            message = f"Invalid {field}."
        break
    return _error(422, message, request)


class TravelRequest(BaseModel):
    message: str = Field(max_length=config.MAX_MESSAGE_CHARS)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)

    @field_validator("thread_id", mode="before")
    @classmethod
    def empty_thread_id_is_new_thread(cls, value):
        return value or None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


def _plan_trip(message: str, thread_id: str | None):
    # Unknown ids are replaced with a fresh server-generated one so clients
    # can't choose (predictable) thread ids.
    if thread_id and not backend.thread_exists(thread_id):
        logger.info("Ignoring unknown thread id %s", thread_ref(thread_id))
        thread_id = None
    return backend.run_travel_agent(user_input=message, thread_id=thread_id)


def _is_timeout(exc: BaseException) -> bool:
    try:
        import groq
        import httpx
        timeout_types = (groq.APITimeoutError, httpx.TimeoutException, TimeoutError)
    except ImportError:  # pragma: no cover
        timeout_types = (TimeoutError,)
    return isinstance(exc, timeout_types)


def _is_database_error(exc: BaseException) -> bool:
    import psycopg
    import psycopg_pool
    return isinstance(exc, (backend.BackendUnavailable, psycopg.OperationalError, psycopg_pool.PoolTimeout))


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.hit(client_ip)
    if not allowed:
        return _error(
            429,
            "Too many requests. Please wait a moment and try again.",
            request,
            headers={"Retry-After": str(retry_after)},
        )

    user_message = request_data.message.strip()

    if not user_message:
        return _error(400, "Message cannot be empty.", request)

    try:
        result = await run_in_threadpool(_plan_trip, user_message, request_data.thread_id)
    except Exception as exc:
        request_id = request.state.request_id
        if _is_database_error(exc):
            logger.warning("Request %s failed: database unavailable (%s)", request_id, type(exc).__name__)
            return _error(503, UNAVAILABLE_MESSAGE, request)
        if _is_timeout(exc):
            logger.warning("Request %s failed: LLM/API timeout (%s)", request_id, type(exc).__name__)
            return _error(504, TIMEOUT_MESSAGE, request)
        logger.exception("Request %s failed with an unexpected error", request_id)
        return _error(500, GENERIC_ERROR_MESSAGE, request)

    return JSONResponse(
        content={
            "success": True,
            "thread_id": result["thread_id"],
            "answer": result["answer"],
            "flight_results": result["flight_results"],
            "hotel_results": result["hotel_results"],
            "itinerary": result["itinerary"],
            "llm_calls": result["llm_calls"],
        }
    )


async def _database_ok(max_age: float) -> bool:
    if not backend.is_backend_ready():
        return False
    now = time.monotonic()
    if now - _db_health["checked_at"] >= max_age:
        _db_health["ok"] = await run_in_threadpool(backend.check_database, 2.0)
        _db_health["checked_at"] = now
    return _db_health["ok"]


@app.get("/health")
async def health_check():
    """Liveness: 200 while the process runs; reports database status."""
    db_ok = await _database_ok(DB_HEALTH_CACHE_SECONDS)
    return {
        "status": "ok" if db_ok else "degraded",
        "message": "AI Travel Planner API is running",
        "checks": {"database": "ok" if db_ok else "unavailable"},
    }


@app.get("/health/ready")
async def readiness_check():
    """Readiness: 200 only when the database and agent graph are usable."""
    if not backend.is_backend_ready():
        try:
            await run_in_threadpool(backend.init_backend)
        except backend.BackendUnavailable:
            pass
    db_ok = await _database_ok(0)
    status_code = 200 if db_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if db_ok else "unavailable",
            "checks": {"database": "ok" if db_ok else "unavailable"},
        },
    )


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
