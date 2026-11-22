import logging
import os
import threading
import time
from typing import TypedDict, Annotated
import operator

import config
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from security import new_thread_id, thread_ref

logger = logging.getLogger("tripmate.backend")

# Render Postgres reserves this many connections for direct/internal use.
RESERVED_DB_CONNECTIONS = 10


class BackendUnavailable(RuntimeError):
    """The database/graph is not ready. Message is safe to log, not to return."""


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise BackendUnavailable(
            "DATABASE_URL is missing. Please add your PostgreSQL connection URL to .env"
        )

    if "sslmode=" not in database_url and config.DB_SSLMODE:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode={config.DB_SSLMODE}"

    return database_url


# =========================
# LLM
# =========================

_llm = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise BackendUnavailable("GROQ_API_KEY is missing. Please add it to your .env file.")
                _llm = ChatGroq(
                    model=config.GROQ_MODEL,
                    api_key=api_key,
                    timeout=config.LLM_TIMEOUT_SECONDS,
                    max_retries=config.LLM_MAX_RETRIES,
                )
    return _llm


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
    }



# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
    }




# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = get_llm().invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = get_llm().invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer (connection pool, lazily initialised)
# =========================

_lock = threading.Lock()
_pool: ConnectionPool | None = None
_checkpointer: PostgresSaver | None = None
_travel_graph = None
_last_init_attempt = 0.0


def _check_connection_budget(pool: ConnectionPool) -> None:
    try:
        with pool.connection() as conn:
            row = conn.execute("SHOW max_connections").fetchone()
        max_connections = int(row["max_connections"])
    except Exception:
        logger.warning("Could not read max_connections from the database")
        return

    needed = config.DB_POOL_MAX * max(1, config.WEB_CONCURRENCY)
    available = max_connections - RESERVED_DB_CONNECTIONS
    if needed > available:
        logger.warning(
            "DB pool may exceed the database connection limit: pool max %d x %d worker(s) = %d, "
            "database allows %d (%d reserved). Lower DB_POOL_MAX.",
            config.DB_POOL_MAX, config.WEB_CONCURRENCY, needed, max_connections,
            RESERVED_DB_CONNECTIONS,
        )
    else:
        logger.info(
            "DB pool: up to %d connection(s) per worker; database max_connections=%d",
            config.DB_POOL_MAX, max_connections,
        )


def init_backend(force: bool = False) -> None:
    """Open the connection pool, create checkpoint tables and compile the graph.

    Safe to call repeatedly; retries are throttled to DB_INIT_RETRY_SECONDS
    unless ``force`` is set. Raises BackendUnavailable on failure.
    """
    global _pool, _checkpointer, _travel_graph, _last_init_attempt

    if _travel_graph is not None:
        return

    # Never queue behind an in-progress attempt (it can take several seconds
    # while the database is down); callers get BackendUnavailable instead.
    if not _lock.acquire(blocking=force):
        raise BackendUnavailable("Database initialisation is in progress; please wait.")
    try:
        if _travel_graph is not None:
            return

        now = time.monotonic()
        if not force and now - _last_init_attempt < config.DB_INIT_RETRY_SECONDS:
            raise BackendUnavailable("Database initialisation is being retried; please wait.")
        _last_init_attempt = now

        pool = None
        try:
            pool = ConnectionPool(
                get_database_url(),
                min_size=config.DB_POOL_MIN,
                max_size=config.DB_POOL_MAX,
                timeout=config.DB_POOL_TIMEOUT_SECONDS,
                max_idle=300,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "connect_timeout": config.DB_CONNECT_TIMEOUT_SECONDS,
                    "application_name": "tripmate",
                },
                check=ConnectionPool.check_connection,
                open=False,
                name="tripmate",
            )
            pool.open(wait=True, timeout=config.DB_POOL_TIMEOUT_SECONDS)

            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            _check_connection_budget(pool)

            _travel_graph = graph.compile(checkpointer=checkpointer)
            _checkpointer = checkpointer
            _pool = pool
            logger.info("Backend ready: database pool open and checkpoint tables verified")
        except BackendUnavailable:
            if pool is not None:
                pool.close()
            raise
        except Exception as exc:
            if pool is not None:
                try:
                    pool.close(timeout=1)
                except Exception:
                    pass
            logger.warning("Database unavailable during initialisation (%s)", type(exc).__name__)
            raise BackendUnavailable("Database is unavailable.") from exc
    finally:
        _lock.release()


def close_backend() -> None:
    """Close the pool and drop the compiled graph (application shutdown)."""
    global _pool, _checkpointer, _travel_graph
    with _lock:
        pool = _pool
        _pool = None
        _checkpointer = None
        _travel_graph = None
    if pool is not None:
        pool.close(timeout=5)
        logger.info("Database pool closed")


def is_backend_ready() -> bool:
    return _travel_graph is not None


def get_graph():
    if _travel_graph is None:
        init_backend()
    return _travel_graph


def check_database(timeout: float = 2.0) -> bool:
    """Cheap connectivity probe through the pool (SELECT 1)."""
    pool = _pool
    if pool is None:
        return False
    try:
        with pool.connection(timeout=timeout) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.warning("Database health check failed (%s)", type(exc).__name__)
        return False


def thread_exists(thread_id: str) -> bool:
    get_graph()
    config_ = {"configurable": {"thread_id": thread_id}}
    return _checkpointer.get_tuple(config_) is not None


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    travel_graph = get_graph()

    if not thread_id:
        thread_id = new_thread_id()

    config_ = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    logger.info("Running travel graph for thread %s", thread_ref(thread_id))

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config_
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
