import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

import backend
from backend import (
    _is_rate_limit_or_token_error,
    _safe_llm_invoke,
    final_agent,
    itinerary_agent,
    budget_agent,
    flight_agent,
    _serialize_result,
    resume_travel_agent,
    TravelState,
)


class MockRateLimitException(Exception):
    def __init__(
        self,
        message="Rate limit reached for model openai/gpt-oss-120b on tokens per minute (TPM): Limit 8,000, Requested 12,358.",
        status_code=413,
    ):
        super().__init__(message)
        self.status_code = status_code


class TestTokenLimitsAndErrorHandling(unittest.TestCase):
    """Test suite for Groq TPM token limits, rate-limit detection, and safe fallback handling."""

    def test_rate_limit_error_detection(self):
        # 413 status code
        e1 = MockRateLimitException("413 Request Entity Too Large", status_code=413)
        self.assertTrue(_is_rate_limit_or_token_error(e1))

        # 429 status code
        e2 = MockRateLimitException("Too Many Requests", status_code=429)
        self.assertTrue(_is_rate_limit_or_token_error(e2))

        # Groq specific TPM message
        e3 = Exception(
            "Rate limit reached for model openai/gpt-oss-120b on tokens per minute (TPM): Limit 8,000, Requested 12,358."
        )
        self.assertTrue(_is_rate_limit_or_token_error(e3))

        # Context length exceeded
        e4 = Exception("Maximum context length exceeded: requested 12000 tokens.")
        self.assertTrue(_is_rate_limit_or_token_error(e4))

        # Generic unrelated error should return False
        e5 = ValueError("Invalid input value")
        self.assertFalse(_is_rate_limit_or_token_error(e5))

    def test_safe_llm_invoke_with_rate_limit_error(self):
        # When LLM raises rate limit exception, _safe_llm_invoke should use fallback without crashing
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = MockRateLimitException()
        mock_llm.bind.return_value = MagicMock(invoke=MagicMock(side_effect=MockRateLimitException()))

        with patch("backend.llm", mock_llm):
            content, is_fallback = _safe_llm_invoke(
                system_prompt="Test system",
                user_prompt="Test user",
                fallback_fn=lambda: "Safe Fallback Content",
                max_tokens=500,
            )
            self.assertTrue(is_fallback)
            self.assertEqual(content, "Safe Fallback Content")

    def test_final_agent_survives_groq_413_tpm_error(self):
        # Simulates the exact crash scenario:
        # User approves draft -> final_agent runs -> Groq returns 413 TPM limit exceeded (12,358 tokens)
        state: TravelState = {
            "user_query": "Plan a 5-day cultural trip to Kyoto for 2 with $3000 budget",
            "trip_constraints": {
                "destination": "Kyoto",
                "duration": "5 days",
                "budget": "$3000",
            },
            "flight_results": "Recommended: ANA direct to KIX. Est: $1000/person.",
            "hotel_results": "Hotel Kanra Kyoto (~$160/night), Sakura Terrace (~$120/night).",
            "weather_results": "Kyoto: 18°C, pleasant autumn weather. Light layers advised.",
            "budget_results": "Trip budget is realistic. Flights $2000, Hotels $700, Daily $300.",
            "itinerary": "# Day 1: Arashiyama Bamboo Grove\n# Day 2: Fushimi Inari & Gion\n# Day 3: Kinkaku-ji",
            "approved": True,
            "human_feedback": "",
            "llm_calls": 4,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = MockRateLimitException()
        mock_llm.bind.return_value = MagicMock(invoke=MagicMock(side_effect=MockRateLimitException()))

        with patch("backend.llm", mock_llm):
            result = final_agent(state)

        # Application must NOT crash and must return final_response
        self.assertIn("final_response", result)
        response_text = result["final_response"]

        # Response must contain all 7 sections
        self.assertIn("1. Trip Summary", response_text)
        self.assertIn("2. Flight Information", response_text)
        self.assertIn("3. Hotel Suggestions", response_text)
        self.assertIn("4. Weather Information", response_text)
        self.assertIn("5. Day-by-Day Itinerary", response_text)
        self.assertIn("6. Estimated Budget", response_text)
        self.assertIn("7. Final Recommendations", response_text)

        # Essential travel details preserved
        self.assertIn("Kyoto", response_text)
        self.assertIn("5 days", response_text)

    def test_final_agent_with_revision_feedback_under_rate_limit(self):
        # Simulates rejection with revision feedback when Groq hits rate limit
        state: TravelState = {
            "user_query": "Plan a 4-day trip to Barcelona",
            "trip_constraints": {"destination": "Barcelona", "duration": "4 days"},
            "flight_results": "Vueling or Iberia. Approx $300.",
            "hotel_results": "Hotel Jazz near Placa Catalunya (~$130/night).",
            "weather_results": "Barcelona: 20°C, sunny.",
            "budget_results": "Budget feasible.",
            "itinerary": "# Day 1: Gothic Quarter\n# Day 2: Sagrada Familia",
            "approved": False,
            "human_feedback": "Please add more beach time and Gaudí architecture.",
            "llm_calls": 4,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = MockRateLimitException()
        mock_llm.bind.return_value = MagicMock(invoke=MagicMock(side_effect=MockRateLimitException()))

        with patch("backend.llm", mock_llm):
            result = final_agent(state)

        response_text = result["final_response"]
        self.assertIn("Barcelona", response_text)
        # Human feedback note must be incorporated
        self.assertIn("more beach time and Gaudí architecture", response_text)
        self.assertIn("5. Day-by-Day Itinerary", response_text)

    def test_itinerary_agent_survives_rate_limit_and_preserves_approval_state(self):
        # If itinerary_agent hits a rate limit before approval, it must NOT crash
        # and must preserve the pending approval request
        state: TravelState = {
            "user_query": "5-day trip to London",
            "trip_constraints": {"destination": "London", "duration": "5 days"},
            "flight_results": "British Airways to LHR.",
            "hotel_results": "Premier Inn London Hub.",
            "weather_results": "London: 14°C, occasional rain.",
            "budget_results": "Estimated cost $1800.",
            "llm_calls": 3,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = MockRateLimitException()
        mock_llm.bind.return_value = MagicMock(invoke=MagicMock(side_effect=MockRateLimitException()))

        with patch("backend.llm", mock_llm):
            result = itinerary_agent(state)

        # Must have generated draft itinerary and approval request
        self.assertIn("itinerary", result)
        self.assertIn("approval_request", result)
        self.assertTrue(len(result["itinerary"]) > 50)
        self.assertIn("London", result["itinerary"])
        self.assertIn("Please review the generated draft itinerary", result["approval_request"])

    def test_budget_and_flight_agents_safe_fallback(self):
        state: TravelState = {
            "user_query": "Trip to Rome",
            "trip_constraints": {"destination": "Rome", "budget": "$2000"},
            "llm_calls": 1,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = MockRateLimitException()
        mock_llm.bind.return_value = MagicMock(invoke=MagicMock(side_effect=MockRateLimitException()))

        with patch("backend.llm", mock_llm):
            with patch("backend.asyncio.run", return_value="[Airport data]"):
                b_res = budget_agent(state)
                f_res = flight_agent(state)

        self.assertIn("budget_results", b_res)
        self.assertIn("Rome", b_res["budget_results"])
        self.assertIn("flight_results", f_res)
        self.assertIn("Flight guidance", f_res["flight_results"])

    def test_prompt_character_and_token_safety_margin(self):
        # Test that the prompt sent to final_agent stays well within the 8,000 TPM limit
        state: TravelState = {
            "user_query": "Plan a 7-day vacation to Paris for 2 adults with $3500 budget",
            "trip_constraints": {
                "destination": "Paris",
                "origin": "New York",
                "duration": "7 days",
                "budget": "$3500",
                "travel_style": "Comfortable",
            },
            "flight_results": "Direct flights JFK to CDG via Delta/Air France ($850-$1100 roundtrip). Book 6 weeks early.",
            "hotel_results": "- **Hotel Le Marais**: Boutique stay in central Paris (~$180/night).\n- **Hotel Malte**: Opera district (~$160/night).",
            "weather_results": "Current in Paris: 17°C, mild. Forecast: 15-18°C with light scattered showers.",
            "budget_results": "Budget breakdown: Flights $1900, Hotel $1200, Food/Activities $400. Feasible with budget tracking.",
            "itinerary": """# Day 1: Arrival & Seine Cruise
# Day 2: Louvre Museum & Tuileries Garden
# Day 3: Eiffel Tower & Champs-Élysées
# Day 4: Montmartre & Sacré-Cœur
# Day 5: Palace of Versailles Day Trip
# Day 6: Latin Quarter & Notre Dame
# Day 7: Souvenir Shopping & Departure""",
            "approved": True,
            "human_feedback": "",
            "llm_calls": 5,
        }

        captured_prompts = []

        def mock_safe_invoke(system_prompt, user_prompt, fallback_fn=None, max_tokens=None):
            captured_prompts.append(user_prompt)
            return "Sample final response", False

        with patch("backend._safe_llm_invoke", side_effect=mock_safe_invoke):
            final_agent(state)

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]

        # Rule of thumb: ~4 characters per token in English prose
        estimated_prompt_tokens = len(prompt) / 4.0

        # Prompt should be under 2,000 tokens (8,000 characters)
        self.assertLess(estimated_prompt_tokens, 2000)

        # Combined prompt tokens + max_tokens (2048) must be well below 8,000 TPM limit
        total_requested_tokens = estimated_prompt_tokens + backend.MAX_COMPLETION_TOKENS
        self.assertLess(total_requested_tokens, 5000)

        # Safety margin is at least 3,000 tokens (> 37.5% safety margin)
        safety_margin = 8000 - total_requested_tokens
        self.assertGreater(safety_margin, 3000)

    def test_serialize_result_contains_all_api_fields(self):
        # Ensure all existing API fields are preserved for frontend compatibility
        sample_result = {
            "final_response": "Polished final answer",
            "flight_results": "Flight info",
            "hotel_results": "Hotel info",
            "weather_results": "Weather info",
            "budget_results": "Budget info",
            "itinerary": "Itinerary info",
            "selected_agents": ["flight_agent", "hotel_agent", "itinerary_agent"],
            "trip_constraints": {"destination": "Tokyo"},
            "supervisor_reasoning": "Reasoning",
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "approved": True,
            "human_feedback": "",
            "llm_calls": 5,
        }

        serialized = _serialize_result(sample_result, "thread_123")

        expected_fields = [
            "thread_id",
            "answer",
            "requires_approval",
            "approval_request",
            "flight_results",
            "hotel_results",
            "weather_results",
            "budget_results",
            "itinerary",
            "selected_agents",
            "trip_constraints",
            "supervisor_reasoning",
            "guardrail_allowed",
            "guardrail_reason",
            "approved",
            "human_feedback",
            "llm_calls",
        ]

        for field in expected_fields:
            self.assertIn(field, serialized)

        self.assertEqual(serialized["thread_id"], "thread_123")
        self.assertEqual(serialized["answer"], "Polished final answer")


if __name__ == "__main__":
    unittest.main()
