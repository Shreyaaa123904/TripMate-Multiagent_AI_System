from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage

from trip_state import StructuredTripState


UNKNOWN_TOKENS = {
    "",
    "unknown",
    "unspecified",
    "not sure",
    "n/a",
    "none",
    "null",
    "na",
    "tbd",
}


class TripExtractionResult(BaseModel):
    state: StructuredTripState | None = None
    clarifications: list[str] = Field(default_factory=list)
    extraction_error: str | None = None
    raw_response: str = ""


def extract_structured_trip(message: str, llm: Any | None = None) -> TripExtractionResult:
    """
    Extract a StructuredTripState from natural language user input.

    This function performs extraction and validation only. It does not merge with
    saved state.
    """
    user_message = (message or "").strip()
    if not user_message:
        return TripExtractionResult(
            state=None,
            clarifications=["Please provide your travel request so I can extract trip details."],
            extraction_error="empty_message",
        )

    model = llm or _default_llm()
    try:
        invoke_model = model.bind(max_tokens=400) if hasattr(model, "bind") else model
        response = invoke_model.invoke(
            [
                SystemMessage(content=_system_prompt()),
                HumanMessage(content=_user_prompt(user_message)),
            ]
        )
        raw = str(getattr(response, "content", response))
    except Exception as exc:
        return TripExtractionResult(
            state=None,
            clarifications=["I could not read your request reliably. Please restate your trip details."],
            extraction_error=f"llm_error: {type(exc).__name__}",
        )

    try:
        payload = _json_from_text(raw)
    except Exception as exc:
        return TripExtractionResult(
            state=None,
            clarifications=["I could not parse your request. Please provide details in a simpler format."],
            extraction_error=f"json_parse_error: {exc}",
            raw_response=raw,
        )

    normalized_payload = _normalize_unknowns(payload)

    try:
        state = StructuredTripState.model_validate(normalized_payload)
    except ValidationError as exc:
        return TripExtractionResult(
            state=None,
            clarifications=_validation_clarifications(exc),
            extraction_error="validation_error",
            raw_response=raw,
        )

    clarifications = _collect_clarifications(normalized_payload, state)

    return TripExtractionResult(
        state=state,
        clarifications=list(dict.fromkeys(clarifications)),
        extraction_error=None,
        raw_response=raw,
    )


def _default_llm() -> Any:
    # Reuse the existing project LLM configuration.
    import backend

    if hasattr(backend, "get_llm"):
        return backend.get_llm()
    return getattr(backend, "llm", None)


def _json_from_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")

    return json.loads(text[start : end + 1])


def _normalize_unknowns(value: Any, field_name: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_unknowns(v, k) for k, v in value.items()}

    if isinstance(value, list):
        return [_normalize_unknowns(v, field_name) for v in value]

    if isinstance(value, str):
        normalized = value.strip()
        # Preserve enum/control tokens where "unknown" is a valid semantic value.
        if field_name in {"granularity", "operation", "reset_scope"}:
            return normalized
        if normalized.lower() in UNKNOWN_TOKENS:
            return None
        return normalized

    return value


def _validation_clarifications(exc: ValidationError) -> list[str]:
    notes: list[str] = [
        "Some extracted trip details were invalid. Please clarify the highlighted fields.",
    ]

    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        if loc:
            notes.append(f"{loc}: {msg}")
        else:
            notes.append(str(msg))

    return notes


def _collect_clarifications(payload: dict[str, Any], state: StructuredTripState) -> list[str]:
    clarifications: list[str] = []

    incoming = payload.get("clarifications")
    if isinstance(incoming, list):
        clarifications.extend(str(item) for item in incoming if str(item).strip())

    if state.destination is None or state.destination.is_ambiguous():
        clarifications.append("Destination is unclear. Please provide country, city, or airport code.")

    if state.travellers is not None and state.travellers.is_ambiguous():
        clarifications.append("Traveller counts are ambiguous. Please provide adults/children/infants and rooms if needed.")

    if state.budget_currency_ambiguous and not state.budget_currency:
        clarifications.append("Budget currency is unclear. Please provide an ISO currency code (e.g. USD, EUR).")

    if state.operation.value in {"remove_fields", "reset"} and not state.remove_fields and not state.reset_scope:
        clarifications.append("Please specify which fields to remove or which reset scope to apply.")

    return clarifications


def _system_prompt() -> str:
    return (
        "You extract structured travel information from user messages. "
        "Return strict JSON only. Never invent unknown values. "
        "Keep unknown fields null and mark ambiguity explicitly."
    )


def _user_prompt(message: str) -> str:
    return f"""
Extract trip intelligence from this user message.

Allowed operation values:
- auto
- new_trip
- update_trip
- follow_up_question
- remove_fields
- reset

Output strict JSON with this schema shape:
{{
  "operation": "auto|new_trip|update_trip|follow_up_question|remove_fields|reset",
  "origin": {{
    "raw_text": "",
    "granularity": "unknown|country|city|airport",
    "country": null,
    "city": null,
    "airport_iata": null
  }},
  "destination": {{
    "raw_text": "",
    "granularity": "unknown|country|city|airport",
    "country": null,
    "city": null,
    "airport_iata": null
  }},
  "dates": {{
    "departure_date": null,
    "return_date": null,
    "duration_days": null
  }},
  "travellers": {{
    "adults": null,
    "children": null,
    "infants": null,
    "rooms": null,
    "ambiguous_text": null
  }},
  "budget_amount": null,
  "budget_currency": null,
  "budget_currency_raw": null,
  "budget_currency_ambiguous": false,
  "preferences": [],
  "remove_fields": [],
  "reset_scope": null,
  "clarifications": []
}}

Rules:
- Do not invent airports, currencies, dates, or traveller counts.
- If not provided, use null.
- If duration in days is mentioned (e.g. "7-day", "5 days"), set dates.duration_days.
- If departure/return calendar dates are mentioned, extract ISO YYYY-MM-DD dates into dates.departure_date and dates.return_date.
- Use granularity=unknown for unresolved locations.
- If currency phrase is ambiguous (e.g. "dollars" without region), set budget_currency_ambiguous=true and budget_currency_raw.
- If the message asks a follow-up question without changing travel parameters, use operation=follow_up_question.
- If the message asks to remove existing fields, use operation=remove_fields with remove_fields.
- If the message asks to reset context, use operation=reset with reset_scope.

User message:
{message}
"""
