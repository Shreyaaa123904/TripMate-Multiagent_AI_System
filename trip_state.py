from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any

import pycountry
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class TripOperation(str, Enum):
    AUTO = "auto"
    NEW_TRIP = "new_trip"
    UPDATE = "update_trip"
    FOLLOW_UP = "follow_up_question"
    REMOVE = "remove_fields"
    RESET = "reset"


class LocationGranularity(str, Enum):
    UNKNOWN = "unknown"
    COUNTRY = "country"
    CITY = "city"
    AIRPORT = "airport"


SUPPORTED_REMOVE_FIELDS = {
    "origin",
    "destination",
    "dates",
    "travellers",
    "budget",
    "preferences",
}

SUPPORTED_RESET_SCOPES = {
    "trip",
    "budget",
    "dates",
    "travellers",
}


class TripLocation(BaseModel):
    raw_text: str | None = None
    granularity: LocationGranularity = LocationGranularity.UNKNOWN
    country: str | None = None
    city: str | None = None
    airport_iata: str | None = None

    @field_validator("airport_iata")
    @classmethod
    def normalize_iata(cls, value: str | None) -> str | None:
        if value is None:
            return value
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("airport_iata must be a 3-letter IATA code")
        return code

    @model_validator(mode="after")
    def validate_granularity(self) -> "TripLocation":
        if self.granularity == LocationGranularity.COUNTRY:
            if not self.country:
                raise ValueError("country-level destination requires country")
            if self.city or self.airport_iata:
                raise ValueError("country-level destination cannot include city or airport_iata")

        if self.granularity == LocationGranularity.CITY:
            if not self.city:
                raise ValueError("city-level destination requires city")
            if self.airport_iata:
                raise ValueError("city-level destination cannot include airport_iata")

        if self.granularity == LocationGranularity.AIRPORT:
            if not self.airport_iata:
                raise ValueError("airport-level destination requires airport_iata")

        return self

    def is_ambiguous(self) -> bool:
        return self.granularity == LocationGranularity.UNKNOWN


class TripDates(BaseModel):
    departure_date: date | None = None
    return_date: date | None = None
    duration_days: int | None = None

    @field_validator("duration_days")
    @classmethod
    def validate_duration_positive(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("duration_days must be positive when provided")
        return value

    @model_validator(mode="after")
    def validate_date_order(self) -> "TripDates":
        if self.departure_date and self.return_date and self.return_date < self.departure_date:
            raise ValueError("return_date cannot be earlier than departure_date")
        return self


class TravellerCounts(BaseModel):
    adults: int | None = None
    children: int | None = None
    infants: int | None = None
    rooms: int | None = None
    ambiguous_text: str | None = None

    @field_validator("adults", "children", "infants")
    @classmethod
    def validate_non_negative(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value < 0:
            raise ValueError("traveller counts must be non-negative")
        return value

    @field_validator("rooms")
    @classmethod
    def validate_rooms_positive(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("rooms must be positive when provided")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> "TravellerCounts":
        numeric_values = [self.adults, self.children, self.infants, self.rooms]
        has_numeric = any(v is not None for v in numeric_values)

        if self.ambiguous_text and has_numeric:
            raise ValueError("ambiguous traveller text cannot be combined with numeric traveller counts")

        if not self.ambiguous_text and not has_numeric:
            raise ValueError("traveller counts require numeric values or ambiguous_text")

        if self.adults is not None and self.adults == 0 and (self.children or self.infants):
            raise ValueError("children/infants without adults are not supported")

        return self

    def is_ambiguous(self) -> bool:
        return bool(self.ambiguous_text)


class StructuredTripState(BaseModel):
    operation: TripOperation = TripOperation.UPDATE
    origin: TripLocation | None = None
    destination: TripLocation | None = None
    dates: TripDates | None = None
    travellers: TravellerCounts | None = None

    budget_amount: Decimal | None = None
    budget_currency: str | None = None
    budget_currency_raw: str | None = None
    budget_currency_ambiguous: bool = False

    preferences: list[str] = Field(default_factory=list)

    remove_fields: list[str] = Field(default_factory=list)
    reset_scope: str | None = None

    @field_validator("budget_currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return value
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("budget_currency must be a 3-letter ISO code")
        if pycountry.currencies.get(alpha_3=code) is None:
            raise ValueError("budget_currency must be a valid ISO 4217 currency code")
        return code

    @field_validator("budget_amount")
    @classmethod
    def validate_budget_amount(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("budget_amount must be positive")
        return value

    @field_validator("remove_fields")
    @classmethod
    def validate_remove_fields(cls, fields: list[str]) -> list[str]:
        unknown = sorted(set(fields) - SUPPORTED_REMOVE_FIELDS)
        if unknown:
            raise ValueError(f"unsupported remove_fields: {', '.join(unknown)}")
        # Keep stable order while deduplicating.
        return list(dict.fromkeys(fields))

    @field_validator("reset_scope")
    @classmethod
    def validate_reset_scope(cls, scope: str | None) -> str | None:
        if scope is None:
            return None
        if scope not in SUPPORTED_RESET_SCOPES:
            raise ValueError(f"unsupported reset_scope: {scope}")
        return scope

    @model_validator(mode="after")
    def validate_operation(self) -> "StructuredTripState":
        if self.operation == TripOperation.REMOVE and not self.remove_fields:
            raise ValueError("remove_fields operation requires at least one supported remove field")

        if self.operation == TripOperation.RESET and not self.reset_scope:
            raise ValueError("reset operation requires a reset_scope")

        if self.operation == TripOperation.NEW_TRIP:
            if self.remove_fields:
                raise ValueError("new_trip operation does not support remove_fields")
            if self.reset_scope:
                raise ValueError("new_trip operation does not support reset_scope")

        return self


class MergeOutcome(BaseModel):
    merged_state: StructuredTripState
    clarifications: list[str] = Field(default_factory=list)
    state_changed: bool = False


class DateUpdateError(ValueError):
    """Raised when newly supplied dates are invalid during merge."""


def validate_new_dates_or_raise(new_dates: TripDates | dict[str, Any]) -> TripDates:
    """Validate new date input independently before it can overwrite saved dates."""
    try:
        return TripDates.model_validate(new_dates)
    except ValidationError as exc:
        raise DateUpdateError(str(exc)) from exc


def merge_trip_state(saved: StructuredTripState | None, incoming: StructuredTripState) -> MergeOutcome:
    """
    Deterministic Python-side merge rules for structured trip state.

    Rules:
    - new_trip creates fresh context.
    - Invalid/ambiguous incoming fields trigger clarification and do not overwrite valid saved values.
    - Newly supplied dates are validated before replacing saved dates.
    - Unsupported operations are rejected by schema validation before merge.
    """
    clarifications: list[str] = []
    state_changed = False
    incoming_fields = set(incoming.model_fields_set)

    base = saved.model_copy(deep=True) if saved else StructuredTripState()

    if incoming.operation == TripOperation.NEW_TRIP:
        base = StructuredTripState()
        state_changed = True

    if incoming.operation == TripOperation.RESET and incoming.reset_scope:
        _apply_reset_scope(base, incoming.reset_scope)
        state_changed = True

    if incoming.operation == TripOperation.REMOVE and incoming.remove_fields:
        _apply_remove_fields(base, incoming.remove_fields)
        state_changed = True

    # Apply regular updates on top of operation-specific behavior.
    if incoming.origin is not None:
        if incoming.origin.is_ambiguous():
            clarifications.append("Origin is ambiguous. Please provide country, city, or airport code.")
        elif base.origin != incoming.origin:
            base.origin = incoming.origin
            state_changed = True

    if incoming.destination is not None:
        if incoming.destination.is_ambiguous():
            clarifications.append("Destination is ambiguous. Please specify country, city, or airport code.")
        elif base.destination != incoming.destination:
            base.destination = incoming.destination
            state_changed = True

    if incoming.dates is not None:
        merged_dates_input = _merge_dates(base.dates, incoming.dates)
        try:
            merged_dates = validate_new_dates_or_raise(merged_dates_input)
        except DateUpdateError:
            clarifications.append("Dates are invalid after update. Please provide a valid departure/return combination.")
        else:
            if base.dates != merged_dates:
                base.dates = merged_dates
                state_changed = True

    if incoming.travellers is not None:
        if incoming.travellers.is_ambiguous():
            clarifications.append("Traveller count is ambiguous. Please provide adults/children/infants.")
        else:
            try:
                merged_travellers = _merge_travellers(base.travellers, incoming.travellers)
            except ValidationError:
                clarifications.append("Traveller counts are invalid after update. Please check adults/children/infants/rooms.")
            else:
                if base.travellers != merged_travellers:
                    base.travellers = merged_travellers
                    state_changed = True

    if incoming.preferences:
        if base.preferences != incoming.preferences:
            base.preferences = list(incoming.preferences)
            state_changed = True

    if incoming.budget_amount is not None and base.budget_amount != incoming.budget_amount:
        base.budget_amount = incoming.budget_amount
        state_changed = True

    definitive_currency_supplied = "budget_currency" in incoming_fields and incoming.budget_currency is not None
    ambiguity_flag_supplied = "budget_currency_ambiguous" in incoming_fields
    raw_ambiguity_supplied = "budget_currency_raw" in incoming_fields

    if definitive_currency_supplied:
        if base.budget_currency != incoming.budget_currency:
            base.budget_currency = incoming.budget_currency
            state_changed = True
        if base.budget_currency_ambiguous:
            base.budget_currency_ambiguous = False
            state_changed = True
        if base.budget_currency_raw is not None:
            base.budget_currency_raw = None
            state_changed = True
    elif ambiguity_flag_supplied and incoming.budget_currency_ambiguous:
        clarifications.append("Budget currency is unclear. Please specify a currency code (e.g. USD, EUR).")
        if not base.budget_currency_ambiguous:
            base.budget_currency_ambiguous = True
            state_changed = True
        if raw_ambiguity_supplied and base.budget_currency_raw != incoming.budget_currency_raw:
            base.budget_currency_raw = incoming.budget_currency_raw
            state_changed = True
    elif ambiguity_flag_supplied and not incoming.budget_currency_ambiguous:
        if base.budget_currency_ambiguous:
            base.budget_currency_ambiguous = False
            state_changed = True
        if base.budget_currency_raw is not None:
            base.budget_currency_raw = None
            state_changed = True
    elif raw_ambiguity_supplied and base.budget_currency_ambiguous:
        if base.budget_currency_raw != incoming.budget_currency_raw:
            base.budget_currency_raw = incoming.budget_currency_raw
            state_changed = True

    base.operation = TripOperation.UPDATE
    return MergeOutcome(merged_state=base, clarifications=clarifications, state_changed=state_changed)


def _apply_reset_scope(state: StructuredTripState, scope: str) -> None:
    if scope == "trip":
        state.origin = None
        state.destination = None
        state.dates = None
        state.travellers = None
        state.budget_amount = None
        state.budget_currency = None
        state.budget_currency_raw = None
        state.budget_currency_ambiguous = False
        state.preferences = []
        return

    if scope == "budget":
        state.budget_amount = None
        state.budget_currency = None
        state.budget_currency_raw = None
        state.budget_currency_ambiguous = False
        return

    if scope == "dates":
        state.dates = None
        return

    if scope == "travellers":
        state.travellers = None


def _apply_remove_fields(state: StructuredTripState, remove_fields: list[str]) -> None:
    for field in remove_fields:
        if field == "origin":
            state.origin = None
        elif field == "destination":
            state.destination = None
        elif field == "dates":
            state.dates = None
        elif field == "travellers":
            state.travellers = None
        elif field == "budget":
            state.budget_amount = None
            state.budget_currency = None
            state.budget_currency_raw = None
            state.budget_currency_ambiguous = False
        elif field == "preferences":
            state.preferences = []


def _merge_dates(saved: TripDates | None, incoming: TripDates) -> dict[str, Any]:
    return {
        "departure_date": (
            incoming.departure_date
            if incoming.departure_date is not None
            else (saved.departure_date if saved else None)
        ),
        "return_date": (
            incoming.return_date
            if incoming.return_date is not None
            else (saved.return_date if saved else None)
        ),
        "duration_days": (
            incoming.duration_days
            if incoming.duration_days is not None
            else (saved.duration_days if saved else None)
        ),
    }


def _merge_travellers(saved: TravellerCounts | None, incoming: TravellerCounts) -> TravellerCounts:
    return TravellerCounts(
        adults=(incoming.adults if incoming.adults is not None else (saved.adults if saved else None)),
        children=(incoming.children if incoming.children is not None else (saved.children if saved else None)),
        infants=(incoming.infants if incoming.infants is not None else (saved.infants if saved else None)),
        rooms=(incoming.rooms if incoming.rooms is not None else (saved.rooms if saved else None)),
    )
