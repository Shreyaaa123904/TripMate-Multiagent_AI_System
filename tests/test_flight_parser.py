import pytest

from tools.flight_tool import parse_route, resolve_location_to_iata


@pytest.mark.parametrize(
    "location, expected",
    [
        ("Bangladesh", "DAC"),
        ("Japan", "NRT"),
        ("Dhaka", "DAC"),
        ("Tokyo", "NRT"),
        ("dac", "DAC"),
        ("new york", "JFK"),
        ("", None),
    ],
)
def test_resolve_location_to_iata(location, expected):
    assert resolve_location_to_iata(location) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Trip to Paris from Mumbai", ("BOM", "CDG")),
        ("Plan a trip from Delhi to Tokyo in December", ("DEL", "NRT")),
        ("Fly from London to New York on Friday", ("LHR", "JFK")),
        ("DAC to NRT flights", ("DAC", "NRT")),
        ("Plan a 7 days Japan trip", ("DAC", "NRT")),
        ("Give me all country flight info.", (None, None)),
    ],
)
def test_parse_route_working_cases(query, expected):
    assert parse_route(query) == expected


# Audit bugs B1–B3 (fixed in Phase 1).
@pytest.mark.parametrize(
    "query, expected",
    [
        # B1: destination named before "from X"
        (
            "Plan a complete 7 days Japan trip from Bangladesh including flights, "
            "hotels and sightseeing under 2 lakhs.",
            ("DAC", "NRT"),
        ),
        (
            "Plan a 7 days Thailand trip from Bangladesh with budget hotels and sightseeing.",
            ("DAC", "BKK"),
        ),
        # B2: trailing clause after the origin
        ("Plan a 5 days Dubai trip from Dhaka with flights, hotels and sightseeing.", ("DAC", "DXB")),
        ("Paris trip from Mumbai, with hotels", ("BOM", "CDG")),
        ("Tokyo holiday from Delhi in December for 2 people", ("DEL", "NRT")),
        # B3: uppercase country names / common words are not IATA codes
        ("Trip from USA to UAE", ("JFK", "DXB")),
        ("THE best trip from DAC to NRT", ("DAC", "NRT")),
        ("Plan a trip to UK from USA", ("JFK", "LHR")),
    ],
)
def test_parse_route_audit_bugs_fixed(query, expected):
    assert parse_route(query) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        # UI quick prompts
        (
            "Plan a complete 7 days Japan trip from Bangladesh including flights, hotels "
            "and sightseeing under 2 lakhs.",
            ("DAC", "NRT"),
        ),
        ("Plan a 5 days Dubai trip from Dhaka with flights, hotels and sightseeing.", ("DAC", "DXB")),
        ("Plan a 7 days Thailand trip from Bangladesh with budget hotels and sightseeing.", ("DAC", "BKK")),
        ("Give me all country flight info.", (None, None)),
        # Other shapes
        ("Flights from Dhaka", ("DAC", None)),
        ("Flights to Singapore", (None, "SIN")),
        ("Show flights from DAC", ("DAC", None)),
    ],
)
def test_parse_route_more_cases(query, expected):
    assert parse_route(query) == expected


@pytest.mark.parametrize("word", ["USA", "UAE", "THE", "AND", "FOR"])
def test_common_uppercase_words_not_direct_iata(word):
    assert resolve_location_to_iata(word) != word


def test_free_text_clause_does_not_fuzzy_match():
    assert resolve_location_to_iata("dhaka with flights hotels and sightseeing") is None


def test_flight_request_error_does_not_leak_key(monkeypatch, caplog):
    import requests

    from tools import flight_tool

    secret = "SECRET_ACCESS_KEY_123"
    monkeypatch.setattr(flight_tool, "API_KEY", secret)

    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            f"Max retries exceeded with url: /v1/flights?access_key={secret}"
        )

    monkeypatch.setattr(flight_tool.requests, "get", boom)
    result = flight_tool.search_flights("DAC to NRT")
    assert secret not in result
    assert secret not in caplog.text
    assert "could not be reached" in result
