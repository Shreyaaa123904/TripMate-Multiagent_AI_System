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


# Known bugs from the audit (B1–B3). Marked strict xfail so they start
# failing loudly once fixed, and must then be moved to the passing cases.
@pytest.mark.xfail(strict=True, reason="B1: destination lost when query uses 'from X' without 'to Y'")
@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "Plan a complete 7 days Japan trip from Bangladesh including flights, "
            "hotels and sightseeing under 2 lakhs.",
            ("DAC", "NRT"),
        ),
        (
            "Plan a 7 days Thailand trip from Bangladesh with budget hotels and sightseeing.",
            ("DAC", "BKK"),
        ),
    ],
)
def test_parse_route_destination_before_from(query, expected):
    assert parse_route(query) == expected


@pytest.mark.xfail(strict=True, reason="B2: trailing clause after origin resolves to wrong airport")
def test_parse_route_trailing_clause():
    query = "Plan a 5 days Dubai trip from Dhaka with flights, hotels and sightseeing."
    assert parse_route(query) == ("DAC", "DXB")


@pytest.mark.xfail(strict=True, reason="B3: uppercase words treated as IATA codes")
def test_parse_route_uppercase_country_aliases():
    assert parse_route("Trip from USA to UAE") == ("JFK", "DXB")
