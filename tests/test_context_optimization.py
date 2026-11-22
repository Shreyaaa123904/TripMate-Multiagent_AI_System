import unittest
import json

from backend import (
    _summarize_text,
    _clean_airport_data,
    _clean_airline_data,
    _reduce_hotel_results,
    _reduce_weather_results,
    _synthesize_final_response,
    _synthesize_draft_itinerary,
    _synthesize_budget_results,
    MAX_COMPLETION_TOKENS,
)


class TestContextOptimization(unittest.TestCase):
    """Test suite for context reduction and prompt optimization in TripMate AI."""

    def test_summarize_text(self):
        # Empty string
        self.assertEqual(_summarize_text(""), "")

        # Short text below threshold remains untouched
        short = "Paris is a wonderful destination."
        self.assertEqual(_summarize_text(short, max_chars=100), short)

        # Long text is cleanly truncated on word boundaries
        long_text = "This is a very long text describing many historical landmarks in the city of Paris France."
        summarized = _summarize_text(long_text, max_chars=40)
        self.assertTrue(len(summarized) <= 43)  # with "..."
        self.assertTrue(summarized.endswith("..."))
        # Words should not be chopped mid-word
        self.assertNotIn("landmark...", summarized)

    def test_clean_airport_data(self):
        raw_airports = [
            {"airport_name": "Charles de Gaulle", "iata_code": "CDG", "city_iata_code": "PAR"},
            {"airport_name": "Orly", "iata_code": "ORY", "city_iata_code": "PAR"},
            {"airport_name": "Beauvais", "iata_code": "BVA", "city_iata_code": "BVA"},
            {"airport_name": "Airport 4", "iata_code": "AP4", "city_iata_code": "XYZ"},
            {"airport_name": "Airport 5", "iata_code": "AP5", "city_iata_code": "XYZ"},
            {"airport_name": "Airport 6", "iata_code": "AP6", "city_iata_code": "XYZ"},
            {"airport_name": "Airport 7 (Extra)", "iata_code": "AP7", "city_iata_code": "XYZ"},
        ]
        cleaned = _clean_airport_data(raw_airports, max_chars=300)
        self.assertIn("CDG", cleaned)
        self.assertIn("ORY", cleaned)
        # Slices to top airports, avoiding massive dumps
        self.assertNotIn("Airport 7", cleaned)
        self.assertTrue(len(cleaned) < 300)

    def test_clean_airline_data(self):
        raw_airlines = [
            {"airline_name": "Air France", "iata_code": "AF"},
            {"airline_name": "Delta Air Lines", "iata_code": "DL"},
            {"airline_name": "United Airlines", "iata_code": "UA"},
            {"airline_name": "British Airways", "iata_code": "BA"},
        ]
        cleaned = _clean_airline_data(raw_airlines, max_chars=200)
        self.assertIn("Air France (AF)", cleaned)
        self.assertIn("Delta Air Lines (DL)", cleaned)
        self.assertTrue(len(cleaned) < 200)

    def test_reduce_hotel_results_tavily_dict(self):
        # Simulate verbose Tavily search result with giant raw_content
        verbose_tavily = {
            "query": "Best hotels for Paris trip",
            "results": [
                {
                    "title": "Hotel Le Marais",
                    "url": "https://example.com/hotel1",
                    "content": "Charming boutique hotel in the vibrant Marais neighborhood near cafes and art galleries. Rates from $180/night.",
                    "raw_content": "A" * 8000,  # 8000 chars of HTML boilerplate
                    "score": 0.98,
                },
                {
                    "title": "Eiffel Grand Palace",
                    "url": "https://example.com/hotel2",
                    "content": "Luxury 5-star hotel with breathtaking direct views of the Eiffel Tower, spa, and fine dining. Rates from $350/night.",
                    "raw_content": "B" * 12000,
                    "score": 0.95,
                },
                {
                    "title": "Latin Quarter Inn",
                    "url": "https://example.com/hotel3",
                    "content": "Budget-friendly hotel near the Sorbonne and Notre Dame, great access to metro. Rates from $110/night.",
                    "raw_content": "C" * 6000,
                    "score": 0.91,
                },
            ],
        }

        reduced = _reduce_hotel_results(verbose_tavily)

        # Must keep essential hotel names, locations, and rates
        self.assertIn("Hotel Le Marais", reduced)
        self.assertIn("Eiffel Grand Palace", reduced)
        self.assertIn("Latin Quarter Inn", reduced)

        # Must NOT include the 26,000 chars of raw_content
        self.assertNotIn("A" * 100, reduced)
        self.assertNotIn("B" * 100, reduced)

        # Total length must be compact (under 1,000 characters)
        self.assertTrue(len(reduced) < 1000)

    def test_reduce_hotel_results_json_string(self):
        json_str = json.dumps({
            "results": [
                {"title": "Boutique Rome", "content": "Central location near Colosseum.", "url": "https://example.com"},
            ]
        })
        reduced = _reduce_hotel_results(json_str)
        self.assertIn("Boutique Rome", reduced)
        self.assertIn("Colosseum", reduced)
        self.assertTrue(len(reduced) < 500)

    def test_reduce_weather_results(self):
        weather_data = {
            "city": "Rome",
            "temperature_c": 22.4,
            "feels_like_c": 21.8,
            "humidity": 55,
            "condition": "scattered clouds",
            "wind_speed": 4.1,
        }
        forecast_data = {
            "city": "Rome",
            "forecast": [
                {"datetime": "2026-11-23 12:00:00", "temperature_c": 23.0, "condition": "few clouds"},
                {"datetime": "2026-11-23 15:00:00", "temperature_c": 21.5, "condition": "clear sky"},
                {"datetime": "2026-11-23 18:00:00", "temperature_c": 18.0, "condition": "clear sky"},
            ],
        }

        reduced = _reduce_weather_results(weather_data, forecast_data)

        # Essential weather details are preserved
        self.assertIn("Rome", reduced)
        self.assertIn("22.4°C", reduced)
        self.assertIn("scattered clouds", reduced)
        self.assertIn("55%", reduced)
        self.assertIn("Forecast:", reduced)

        # Total length is under 300 characters (~75 tokens)
        self.assertTrue(len(reduced) < 300)

    def test_completion_tokens_limit_configured(self):
        # Verifies that MAX_COMPLETION_TOKENS is explicitly configured to prevent Groq reserving 4096-8192 tokens
        self.assertLessEqual(MAX_COMPLETION_TOKENS, 2048)

    def test_synthesized_final_response_has_all_seven_sections(self):
        state = {
            "user_query": "7-day trip to Tokyo on a $2500 budget",
            "trip_constraints": {
                "destination": "Tokyo",
                "origin": "San Francisco",
                "duration": "7 days",
                "budget": "$2500",
            },
            "flight_results": "Direct flights SFO to NRT with United or ANA, approx $850-$1100.",
            "hotel_results": "Hotel Gracery Shinjuku (~$140/night), Candeo Hotels Ueno (~$120/night).",
            "weather_results": "Current: 16°C, clear sky. Mild autumn temperatures.",
            "budget_results": "Feasible for 7 days with mid-range dining and public transit.",
            "itinerary": "# Day 1: Shibuya & Shinjuku\n# Day 2: Asakusa & Akihabara\n# Day 3: Meiji Shrine",
            "approved": True,
            "human_feedback": "",
        }

        final_resp = _synthesize_final_response(state)

        # Check all 7 sections are present
        self.assertIn("1. Trip Summary", final_resp)
        self.assertIn("2. Flight Information", final_resp)
        self.assertIn("3. Hotel Suggestions", final_resp)
        self.assertIn("4. Weather Information", final_resp)
        self.assertIn("5. Day-by-Day Itinerary", final_resp)
        self.assertIn("6. Estimated Budget", final_resp)
        self.assertIn("7. Final Recommendations", final_resp)

        # Check essential trip details
        self.assertIn("Tokyo", final_resp)
        self.assertIn("7 days", final_resp)
        self.assertIn("$2500", final_resp)


if __name__ == "__main__":
    unittest.main()
