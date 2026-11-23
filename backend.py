import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search
from trip_extraction import TripExtractionResult, extract_structured_trip
from trip_state import StructuredTripState, TripOperation, merge_trip_state


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM - original model kept, completion tokens capped for TPM safety
# =========================
MAX_COMPLETION_TOKENS = 2048

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=MAX_COMPLETION_TOKENS,
)


def get_llm():
    return llm

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    # Structured trip state intelligence
    structured_trip_state: dict[str, Any] | None
    structured_trip_clarifications: list[str]
    structured_trip_extraction_error: str
    structured_trip_provenance: dict[str, str]
    structured_trip_last_operation: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _summarize_text(text: str, max_chars: int = 500) -> str:
    """Clean and summarize text to a maximum character count without cutting words."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + "..."


def _clean_airport_data(airports: Any, max_chars: int = 400) -> str:
    """Format airport data into a compact list of names, codes, and cities."""
    if not airports:
        return "Airport information not available."
    if isinstance(airports, list):
        items = []
        for a in airports[:6]:
            if isinstance(a, dict):
                name = a.get("airport_name") or a.get("name") or a.get("iata_code", "")
                iata = a.get("iata_code", "")
                city = a.get("city_iata_code") or a.get("country_name", "")
                items.append(f"{name} ({iata}) - {city}" if iata else f"{name}")
            else:
                items.append(str(a))
        return "; ".join(items)
    return _summarize_text(str(airports), max_chars)


def _clean_airline_data(airlines: Any, max_chars: int = 400) -> str:
    """Format airline data into a compact list of airlines."""
    if not airlines:
        return "Airline information not available."
    if isinstance(airlines, list):
        items = []
        for a in airlines[:6]:
            if isinstance(a, dict):
                name = a.get("airline_name") or a.get("name", "")
                iata = a.get("iata_code", "")
                items.append(f"{name} ({iata})" if iata else f"{name}")
            else:
                items.append(str(a))
        return "; ".join(items)
    return _summarize_text(str(airlines), max_chars)


def _reduce_hotel_results(hotel_results: Any, max_hotels: int = 4) -> str:
    """
    Condense raw Tavily search results into essential hotel options.
    Extracts hotel names, neighborhoods, price/ratings, and links.
    Drops raw content, massive JSON blobs, and boilerplate.
    """
    if not hotel_results:
        return "No hotel results available."

    # Parse JSON string if needed
    if isinstance(hotel_results, str):
        trimmed = hotel_results.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            try:
                hotel_results = json.loads(trimmed)
            except Exception:
                pass

    if isinstance(hotel_results, dict):
        raw_items = hotel_results.get("results", [])
    elif isinstance(hotel_results, list):
        raw_items = hotel_results
    else:
        raw_items = None

    if raw_items and isinstance(raw_items, list):
        condensed = []
        for item in raw_items[:max_hotels]:
            if isinstance(item, dict):
                title = item.get("title", "Hotel Option").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "").strip()
                clean_snippet = _summarize_text(content, max_chars=180)
                entry = f"- **{title}**: {clean_snippet}"
                if url:
                    entry += f" ([Info]({url}))"
                condensed.append(entry)
            else:
                condensed.append(f"- {_summarize_text(str(item), max_chars=180)}")
        return "\n".join(condensed)

    return _summarize_text(str(hotel_results), max_chars=800)


def _reduce_weather_results(weather_data: Any, forecast_data: Any) -> str:
    """Condense weather and forecast data into a concise summary (~200 chars)."""
    parts = []
    if isinstance(weather_data, dict):
        city = weather_data.get("city", "Destination")
        temp = weather_data.get("temperature_c", "N/A")
        feels_like = weather_data.get("feels_like_c", "N/A")
        cond = weather_data.get("condition", "N/A")
        humidity = weather_data.get("humidity", "N/A")
        wind = weather_data.get("wind_speed", "N/A")
        parts.append(f"Current in {city}: {temp}°C (feels like {feels_like}°C), {cond}, {humidity}% humidity, wind {wind} m/s.")
    elif weather_data:
        parts.append(f"Current: {_summarize_text(str(weather_data), max_chars=150)}")

    if isinstance(forecast_data, dict) and "forecast" in forecast_data:
        f_items = forecast_data.get("forecast", [])
        f_summaries = []
        for f in f_items[:4]:
            if isinstance(f, dict):
                dt = f.get("datetime", "").split(" ")[-1] if " " in f.get("datetime", "") else f.get("datetime", "")
                f_summaries.append(f"{dt}: {f.get('temperature_c', '')}°C ({f.get('condition', '')})")
        if f_summaries:
            parts.append("Forecast: " + ", ".join(f_summaries) + ".")
    elif forecast_data:
        parts.append(f"Forecast: {_summarize_text(str(forecast_data), max_chars=150)}")

    return "\n".join(parts) if parts else "Weather data unavailable."


def _is_rate_limit_or_token_error(exc: Exception) -> bool:
    """Check if an exception is caused by token limits, TPM limits, or 413/429 status."""
    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    if status_code in (413, 429):
        return True
    indicators = [
        "rate_limit",
        "rate limit",
        "413",
        "429",
        "tpm",
        "tokens per minute",
        "token_limit",
        "token limit",
        "context length",
        "maximum context",
        "requested tokens",
        "payload too large",
        "request too large",
    ]
    return any(ind in msg for ind in indicators)


def _safe_llm_invoke(
    system_prompt: str,
    user_prompt: str,
    fallback_fn=None,
    max_tokens: int | None = None,
) -> tuple[str, bool]:
    """
    Safely invoke the LLM with error handling for token limits and rate limits.
    Returns (response_text, is_fallback).
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        if max_tokens:
            invoker = llm.bind(max_tokens=max_tokens)
            response = invoker.invoke(messages)
        else:
            response = llm.invoke(messages)
        return str(response.content), False
    except Exception as exc:
        print(f"Warning: Primary LLM call failed: {type(exc).__name__}: {exc}")

        if _is_rate_limit_or_token_error(exc):
            try:
                print("Attempting compressed retry under reduced token constraints...")
                compressed_prompt = _summarize_text(user_prompt, max_chars=len(user_prompt) // 2)
                retry_messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=compressed_prompt),
                ]
                invoker = llm.bind(max_tokens=min(max_tokens or 1024, 1024))
                response = invoker.invoke(retry_messages)
                return str(response.content), False
            except Exception as retry_exc:
                print(f"Compressed retry also failed: {retry_exc}")

        if fallback_fn is not None:
            try:
                return fallback_fn(), True
            except Exception as fb_exc:
                print(f"Fallback generation failed: {fb_exc}")

        return (
            "Service temporarily unavailable due to capacity limits. "
            "Please verify your travel details and try again.",
            True,
        )


def _synthesize_budget_results(state: TravelState) -> str:
    constraints = state.get("trip_constraints", {})
    dest = constraints.get("destination") or "your destination"
    budget = constraints.get("budget") or "moderate"
    duration = constraints.get("duration") or "the trip"
    return f"""Estimated Budget Analysis for {dest} ({duration}):
- Budget Category: {budget}
- Flight Allocation: ~35-45% of total budget
- Accommodation: ~30-35% of total budget
- Food & Daily Expenses: ~15-20% of total budget
- Activities & Transit: ~10% of total budget
- Budget Feasibility: Feasible with mindful spending. Consider booking accommodations early and using public transit to stay within budget."""


def _synthesize_draft_itinerary(state: TravelState) -> str:
    query = state.get("user_query", "")
    constraints = state.get("trip_constraints", {})
    dest = constraints.get("destination") or "your destination"
    duration = constraints.get("duration") or "a multi-day stay"
    return f"""# Draft Itinerary: {dest} ({duration})

## Overview
Trip plan for: {query}

### Day 1: Arrival & Exploration
- Morning/Afternoon: Arrive at {dest}, transfer to hotel, check in and freshen up.
- Evening: Take a relaxed orientation stroll around the central neighborhood and enjoy local dining.

### Day 2: Iconic Highlights & Culture
- Morning: Visit top historic landmarks and cultural sights.
- Afternoon: Explore local markets and iconic viewpoints.
- Evening: Traditional dinner and evening leisure.

### Day 3: Local Experiences & Departure Prep
- Morning: Discover hidden gems, neighborhood cafes, and scenic walks.
- Afternoon: Leisurely souvenir shopping and relaxing activity.
- Evening: Farewell dinner celebrating your stay.

*This draft itinerary is ready for your review and approval.*"""


def _synthesize_final_response(state: TravelState) -> str:
    query = state.get("user_query", "")
    constraints = state.get("trip_constraints", {})
    dest = constraints.get("destination") or "Your Destination"
    duration = constraints.get("duration") or "Standard Stay"
    budget = constraints.get("budget") or "Standard"
    approved = state.get("approved", True)
    feedback = state.get("human_feedback", "")

    review_note = ""
    if not approved and feedback:
        review_note = f"\n*Note: Plan adjusted to reflect your revision feedback: '{feedback}'*\n"
    elif approved:
        review_note = "\n*Note: Travel plan finalized based on your approved draft itinerary.*\n"

    itinerary_text = state.get("itinerary", "")
    if not itinerary_text:
        itinerary_text = _synthesize_draft_itinerary(state)

    flight_text = state.get("flight_results") or "Flight options available upon departure booking confirmation."
    hotel_text = state.get("hotel_results") or "Recommended central hotels with good transit access."
    weather_text = state.get("weather_results") or "Pleasant seasonal conditions expected."
    budget_text = state.get("budget_results") or f"Estimated trip budget: {budget}."

    return f"""# Comprehensive Travel Plan: {dest}
{review_note}
## 1. Trip Summary
- **Destination:** {dest}
- **Duration:** {duration}
- **Budget Level:** {budget}
- **Original Request:** {query}

## 2. Flight Information
{flight_text}

## 3. Hotel Suggestions
{hotel_text}

## 4. Weather Information
{weather_text}

## 5. Day-by-Day Itinerary
{itinerary_text}

## 6. Estimated Budget
{budget_text}

## 7. Final Recommendations
- Book flights and accommodations in advance to secure the best rates.
- Keep valid identification, passports, and travel insurance details easily accessible.
- Check current local weather forecasts before packing.
- Use public transit or official rideshare services for convenient local travel."""


def _llm_text(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    content, _ = _safe_llm_invoke(system_prompt, user_prompt, max_tokens=max_tokens)
    return content


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def _constraints_from_structured_state(structured: StructuredTripState | None) -> dict[str, Any]:
    constraints = _empty_constraints()
    if not structured:
        return constraints

    if structured.destination:
        dest = structured.destination.city or structured.destination.country or structured.destination.raw_text or ""
        if dest:
            constraints["destination"] = dest

    if structured.origin:
        orig = structured.origin.city or structured.origin.country or structured.origin.raw_text or ""
        if orig:
            constraints["origin"] = orig

    if structured.dates:
        if structured.dates.duration_days:
            constraints["duration"] = f"{structured.dates.duration_days} days"
        elif structured.dates.departure_date and structured.dates.return_date:
            constraints["duration"] = f"{structured.dates.departure_date} to {structured.dates.return_date}"
        elif structured.dates.departure_date:
            constraints["duration"] = f"From {structured.dates.departure_date}"

    if structured.budget_amount is not None:
        curr = structured.budget_currency or "$"
        constraints["budget"] = f"{curr} {structured.budget_amount}"

    if structured.travellers:
        parts = []
        if structured.travellers.adults is not None:
            parts.append(f"{structured.travellers.adults} adult{'s' if structured.travellers.adults != 1 else ''}")
        if structured.travellers.children is not None:
            parts.append(f"{structured.travellers.children} child{'ren' if structured.travellers.children != 1 else ''}")
        if structured.travellers.rooms is not None:
            parts.append(f"{structured.travellers.rooms} room{'s' if structured.travellers.rooms != 1 else ''}")
        if parts:
            constraints["travel_style"] = ", ".join(parts)

    if structured.preferences:
        constraints["special_preferences"] = list(structured.preferences)

    return constraints


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
            max_tokens=150,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    accumulated_constraints = state.get("trip_constraints", {})
    provenance = state.get("structured_trip_provenance", {})
    changed_fields = [k for k, v in provenance.items() if v == "merged"]
    changed_desc = ", ".join(changed_fields) if changed_fields else "None"

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Accumulated Trip Constraints:
{json.dumps(accumulated_constraints, indent=2)}

Recently Changed Fields:
{changed_desc}

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
            max_tokens=350,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        # Seed base constraints from structured trip state or existing state constraints
        raw_structured = state.get("structured_trip_state")
        base_constraints = _empty_constraints()
        if raw_structured:
            try:
                structured_obj = StructuredTripState.model_validate(raw_structured)
                base_constraints = _constraints_from_structured_state(structured_obj)
            except Exception:
                pass
        if not any(base_constraints.values()):
            base_constraints = dict(state.get("trip_constraints") or _empty_constraints())

        constraints = base_constraints.copy()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            for k, v in parsed_constraints.items():
                if v:
                    constraints[k] = v

        # If provenance indicates destination or budget mutated, ensure relevant specialists run
        provenance = state.get("structured_trip_provenance", {})
        if provenance.get("destination") == "merged":
            for agent_name in ["flight_agent", "hotel_agent", "weather_agent"]:
                if agent_name not in selected_agents:
                    selected_agents.append(agent_name)
        if provenance.get("budget_amount") == "merged" or provenance.get("budget_currency") == "merged":
            if "budget_agent" not in selected_agents:
                selected_agents.append("budget_agent")

        selected_agents = [name for name in AGENT_ORDER if name in selected_agents]
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = dict(state.get("trip_constraints") or _empty_constraints())
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - original behavior kept
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        airport_data = _clean_airport_data(airports)
        airline_data = _clean_airline_data(airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=airport_data,
            airline_data=airline_data,
        )

        content, _ = _safe_llm_invoke(
            system_prompt="You are an expert travel flight planner. Provide concise flight guidance.",
            user_prompt=prompt,
            fallback_fn=lambda: (
                f"Flight guidance for {query}: Major international airlines provide direct or connecting routes. "
                "Book 4-8 weeks in advance for competitive airfares. Check baggage allowances and airport transfer options."
            ),
            max_tokens=400,
        )
        flight_data = content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept, tool output reduced
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results_raw = asyncio.run(
            tavily_mcp_search(query)
        )
        hotel_results = _reduce_hotel_results(hotel_results_raw)

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "General guidance: Select centrally located accommodations near major attractions "
            "or transit lines with verified guest ratings and flexible cancellation."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - original behavior kept, tool output reduced
# =========================
def weather_agent(state: TravelState):
    # Use destination already parsed by supervisor to avoid redundant LLM calls
    destination = state.get("trip_constraints", {}).get("destination", "").strip()
    if not destination:
        try:
            city = extract_destination(state["user_query"])
        except Exception:
            city = state["user_query"]
    else:
        city = destination

    try:
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )
        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )
        weather_results = _reduce_weather_results(weather_data, forecast_data)

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Pack versatile seasonal layers "
            "and check the 5-day forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - context optimized & safe fallback
# =========================
def budget_agent(state: TravelState):
    flight_summary = _summarize_text(state.get("flight_results", ""), max_chars=400)
    hotel_summary = _summarize_text(state.get("hotel_results", ""), max_chars=400)
    weather_summary = _summarize_text(state.get("weather_results", ""), max_chars=250)

    prompt = f"""Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Highlights:
{flight_summary}

Hotel Highlights:
{hotel_summary}

Weather Outlook:
{weather_summary}

Return concise guidance (under 250 words):
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    content, _ = _safe_llm_invoke(
        system_prompt="You are a practical travel budget analyst. Provide concise, realistic budget assessments.",
        user_prompt=prompt,
        fallback_fn=lambda: _synthesize_budget_results(state),
        max_tokens=400,
    )

    return {
        "budget_results": content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - context optimized & safe fallback
# =========================
def itinerary_agent(state: TravelState):
    flight_summary = _summarize_text(state.get("flight_results", ""), max_chars=400)
    hotel_summary = _summarize_text(state.get("hotel_results", ""), max_chars=400)
    weather_summary = _summarize_text(state.get("weather_results", ""), max_chars=250)
    budget_summary = _summarize_text(state.get("budget_results", ""), max_chars=350)

    prompt = f"""Create a structured draft travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Highlights:
{flight_summary}

Hotel Highlights:
{hotel_summary}

Weather Outlook:
{weather_summary}

Budget Analysis:
{budget_summary}

Instructions:
- Provide a clear day-by-day plan with morning, afternoon, and evening activities.
- Keep the plan practical, budget-aware, and aligned with user constraints.
- Create a well-structured draft ready for traveler review and approval.
"""

    content, _ = _safe_llm_invoke(
        system_prompt="You are an expert travel planner. Create realistic, structured draft itineraries.",
        user_prompt=prompt,
        fallback_fn=lambda: _synthesize_draft_itinerary(state),
        max_tokens=1500,
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - minimal context & safe error handling
# =========================
def final_agent(state: TravelState):
    approved = state.get("approved", False)
    feedback = state.get("human_feedback", "").strip()

    if approved:
        review_instruction = (
            "The user approved the draft itinerary. Preserve its schedule, activities, and decisions "
            "while polishing the final presentation."
        )
    else:
        review_instruction = (
            f"The user requested revisions with this feedback: '{feedback}'. "
            "Incorporate these changes into the schedule and finalize the plan."
        )

    # Use concise highlights instead of full duplicate raw dumps
    flight_summary = _summarize_text(state.get("flight_results", ""), max_chars=400)
    hotel_summary = _summarize_text(state.get("hotel_results", ""), max_chars=400)
    weather_summary = _summarize_text(state.get("weather_results", ""), max_chars=250)
    budget_summary = _summarize_text(state.get("budget_results", ""), max_chars=400)
    draft_itinerary = _summarize_text(state.get("itinerary", ""), max_chars=3000)

    final_prompt = f"""Generate the final travel response for the user.

Human Review Status:
{review_instruction}

User Request:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Highlights:
{flight_summary}

Hotel Highlights:
{hotel_summary}

Weather Highlights:
{weather_summary}

Budget Highlights:
{budget_summary}

Draft Itinerary to Polish/Finalize:
{draft_itinerary}

Format the final answer beautifully using these exact 7 sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important guidelines:
- Present clear, practical details in each section.
- Incorporate human feedback when revision was requested.
- Maintain a helpful, professional travel booking tone.
- Keep the response concise and complete without extraneous filler.
"""

    content, _ = _safe_llm_invoke(
        system_prompt="You are a professional AI travel booking assistant. Return a clean, structured travel plan.",
        user_prompt=final_prompt,
        fallback_fn=lambda: _synthesize_final_response(state),
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    response = AIMessage(content=content)

    return {
        "final_response": content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
try:
    _conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
        connect_timeout=3,
    )
    checkpointer = PostgresSaver(_conn)
    checkpointer.setup()
except Exception as exc:
    print(f"PostgreSQL connection unavailable ({exc}). Using MemorySaver fallback.")
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()

travel_graph = graph.compile(checkpointer=checkpointer)
_travel_graph = None


def get_graph():
    global _travel_graph, travel_graph
    if _travel_graph is not None:
        return _travel_graph
    return travel_graph


def _load_saved_structured_trip_state(graph_instance, config_: dict[str, Any]) -> StructuredTripState | None:
    """Load the previously checkpointed structured trip state if present."""
    try:
        snapshot = graph_instance.get_state(config_)
    except Exception:
        return None

    values = getattr(snapshot, "values", {}) or {}
    raw_state = values.get("structured_trip_state")
    if not raw_state:
        return None

    try:
        return StructuredTripState.model_validate(raw_state)
    except Exception:
        return None


def _has_mutating_updates(state: StructuredTripState) -> bool:
    return any(
        [
            bool(state.origin),
            bool(state.destination),
            bool(state.dates),
            bool(state.travellers),
            state.budget_amount is not None,
            state.budget_currency is not None,
            state.budget_currency_raw is not None,
            state.budget_currency_ambiguous,
            bool(state.preferences),
            bool(state.remove_fields),
            bool(state.reset_scope),
        ]
    )


def _build_structured_provenance(
    saved_state: StructuredTripState | None,
    merged_state: StructuredTripState | None,
    extracted_state: StructuredTripState | None,
) -> dict[str, str]:
    fields = [
        "origin",
        "destination",
        "dates",
        "travellers",
        "budget_amount",
        "budget_currency",
        "budget_currency_raw",
        "budget_currency_ambiguous",
        "preferences",
    ]

    provenance: dict[str, str] = {}
    for field in fields:
        extracted_val = getattr(extracted_state, field, None) if extracted_state else None
        saved_val = getattr(saved_state, field, None) if saved_state else None
        merged_val = getattr(merged_state, field, None) if merged_state else None

        if extracted_state is None:
            provenance[field] = "saved"
        elif extracted_val is None and merged_val == saved_val:
            provenance[field] = "saved"
        elif merged_val != saved_val:
            provenance[field] = "merged"
        else:
            provenance[field] = "saved"

    return provenance


def _integrate_structured_trip(
    message: str,
    saved_state: StructuredTripState | None,
) -> tuple[StructuredTripState | None, list[str], str | None, dict[str, str], str]:
    """Extract and safely merge structured trip intelligence for the current message."""
    extraction: TripExtractionResult = extract_structured_trip(message)

    if extraction.state is None:
        # Extraction failed: preserve prior structured state and return clarifications.
        provenance = _build_structured_provenance(saved_state, saved_state, extraction.state)
        return (
            saved_state,
            list(dict.fromkeys(extraction.clarifications)),
            extraction.extraction_error,
            provenance,
            "auto",
        )

    operation = extraction.state.operation.value

    # Follow-up questions should not mutate state unless explicit updates are present.
    if extraction.state.operation == TripOperation.FOLLOW_UP and not _has_mutating_updates(extraction.state):
        provenance = _build_structured_provenance(saved_state, saved_state, extraction.state)
        return (
            saved_state,
            list(dict.fromkeys(extraction.clarifications)),
            extraction.extraction_error,
            provenance,
            operation,
        )

    merge_outcome = merge_trip_state(saved_state, extraction.state)
    combined_clarifications = list(dict.fromkeys(extraction.clarifications + merge_outcome.clarifications))
    provenance = _build_structured_provenance(saved_state, merge_outcome.merged_state, extraction.state)

    return (
        merge_outcome.merged_state,
        combined_clarifications,
        extraction.extraction_error,
        provenance,
        operation,
    )


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
        "structured_trip_state": result.get("structured_trip_state"),
        "structured_trip_clarifications": result.get("structured_trip_clarifications", []),
        "structured_trip_extraction_error": result.get("structured_trip_extraction_error", ""),
        "structured_trip_provenance": result.get("structured_trip_provenance", {}),
        "structured_trip_last_operation": result.get("structured_trip_last_operation", ""),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}
    graph_to_use = get_graph()

    saved_structured_state = _load_saved_structured_trip_state(graph_to_use, config)
    (
        merged_structured_state,
        structured_clarifications,
        structured_extraction_error,
        structured_provenance,
        structured_operation,
    ) = _integrate_structured_trip(user_input, saved_structured_state)

    initial_constraints = _constraints_from_structured_state(merged_structured_state)

    dest_changed = structured_provenance.get("destination") == "merged"
    snapshot = graph_to_use.get_state(config)
    existing_values = getattr(snapshot, "values", {}) or {}

    invoke_payload: dict[str, Any] = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": initial_constraints,
        "supervisor_reasoning": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": 0,
        "structured_trip_state": (
            merged_structured_state.model_dump(mode="json")
            if merged_structured_state is not None
            else None
        ),
        "structured_trip_clarifications": structured_clarifications,
        "structured_trip_extraction_error": structured_extraction_error or "",
        "structured_trip_provenance": structured_provenance,
        "structured_trip_last_operation": structured_operation,
    }

    if not existing_values:
        # Initial turn: initialize all specialist and output channels
        invoke_payload.update({
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
        })
    elif dest_changed and saved_structured_state and saved_structured_state.destination:
        # Destination mutated: invalidate previous flight/hotel/weather results so fresh data is fetched
        invoke_payload.update({
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
        })

    result = graph_to_use.invoke(invoke_payload, config=config)

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    graph_to_use = get_graph()
    try:
        result = graph_to_use.invoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback.strip(),
                }
            ),
            config=config,
        )
        return _serialize_result(result, thread_id)
    except Exception as exc:
        print(f"Error resuming travel agent thread {thread_id}: {exc}")
        # Safeguard: If resuming crashes (e.g. rate limit error or unexpected failure),
        # recover the interrupted state from the checkpointer so we do not lose
        # the pending approval state or return a 500 crash to the user.
        try:
            checkpoint = graph_to_use.get_state(config)
            if checkpoint and checkpoint.values:
                state = dict(checkpoint.values)
                state["approved"] = approved
                state["human_feedback"] = feedback.strip()
                if not state.get("final_response"):
                    state["final_response"] = _synthesize_final_response(state)
                return _serialize_result(state, thread_id)
        except Exception as recovery_exc:
            print(f"State recovery also failed: {recovery_exc}")
        raise