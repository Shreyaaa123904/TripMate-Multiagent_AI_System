import re
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

import backend
import mcp_client
from trip_extraction import extract_structured_trip


def _make_detailed_itinerary(days: int, dest: str = "Kyoto") -> str:
    """Create a realistic >3,000-char multi-day itinerary with meaningful activities."""
    sections = [f"# Draft Itinerary: {dest} ({days} days)\n"]
    for d in range(1, days + 1):
        sections.append(
            f"### Day {d}: {dest} Highlights Part {d}\n"
            f"- **Morning:** Visit Landmark {d}A in {dest}, explore historic streets and temples, and stop at Coffee House {d} for a relaxed local breakfast before peak crowds arrive.\n"
            f"- **Afternoon:** Tour Cultural Museum {d}B, browse Market Square {d}C for regional crafts and snacks, and enjoy an authentic regional lunch near Central Station {d}.\n"
            f"- **Evening:** Sunset promenade walk along Riverfront {d}D followed by a sit-down dinner at Traditional Kitchen {d}E featuring seasonal local specialties."
        )
    return "\n\n".join(sections)


def _assert_all_days_have_meaningful_activities(test_case: unittest.TestCase, text: str, expected_days: int):
    for d in range(1, expected_days + 1):
        pattern = rf"###?\s*Day\s*{d}\b[^\n]*\n([\s\S]*?)(?=\n###?\s*Day\s*\d+\b|\n##\s*\d+\.|\Z)"
        match = re.search(pattern, text, re.IGNORECASE)
        test_case.assertIsNotNone(match, f"Missing Day {d} section in output")
        body = match.group(1).strip()
        # Must not be a bare heading; check for multiple meaningful activity lines
        activity_lines = [ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("*This draft")]
        test_case.assertGreaterEqual(len(activity_lines), 2, f"Day {d} lacks meaningful activities: {body}")
        test_case.assertGreater(len(body), 60, f"Day {d} content too short: {body}")


class TestItineraryCompletenessFixes(unittest.TestCase):
    def test_reasoning_effort_configured_low(self):
        self.assertEqual(backend.llm.reasoning_effort, "low")
        self.assertEqual(mcp_client.llm.reasoning_effort, "low")
        self.assertEqual(backend.llm._default_params.get("reasoning_effort"), "low")

    def test_trip_extraction_binds_max_tokens_400(self):
        mock_llm = MagicMock()
        bound_llm = MagicMock()
        bound_llm.invoke.return_value = AIMessage(
            content='{"operation":"new_trip","destination":{"raw_text":"Japan","granularity":"country","country":"Japan"},"dates":{"duration_days":7},"clarifications":[]}'
        )
        mock_llm.bind.return_value = bound_llm

        res = extract_structured_trip("Plan a 7-day trip to Japan", llm=mock_llm)

        mock_llm.bind.assert_called_once_with(max_tokens=400)
        bound_llm.invoke.assert_called_once()
        self.assertIsNotNone(res.state)
        self.assertEqual(res.state.dates.duration_days, 7)

    def test_extract_requested_days_from_various_sources(self):
        self.assertEqual(
            backend._extract_requested_days({"structured_trip_state": {"dates": {"duration_days": 7}}}),
            7,
        )
        self.assertEqual(
            backend._extract_requested_days(
                {"structured_trip_state": {"dates": {"departure_date": "2027-04-01", "return_date": "2027-04-07"}}}
            ),
            7,
        )
        self.assertEqual(
            backend._extract_requested_days({"trip_constraints": {"duration": "5 days"}}),
            5,
        )
        self.assertEqual(
            backend._extract_requested_days({"user_query": "Plan a 3-day trip to Dubai"}),
            3,
        )

    def test_dynamic_fallback_respects_requested_duration(self):
        for days in (3, 7, 10):
            state = {
                "user_query": f"Plan a {days}-day trip to Kyoto",
                "trip_constraints": {"destination": "Kyoto", "duration": f"{days} days"},
                "structured_trip_state": {"dates": {"duration_days": days}},
            }
            draft = backend._synthesize_draft_itinerary(state)
            _assert_all_days_have_meaningful_activities(self, draft, days)

    def test_safe_llm_invoke_retry_preserves_markdown_and_instructions(self):
        rate_err = Exception(
            "Rate limit reached for model openai/gpt-oss-120b on tokens per minute (TPM): Limit 8,000, Requested 9,100."
        )
        bound_primary = MagicMock()
        bound_primary.invoke.side_effect = rate_err

        bound_retry = MagicMock()
        bound_retry.invoke.return_value = AIMessage(content="### Day 1: Recovered\n- **Morning:** Activity")

        mock_llm = MagicMock()
        mock_llm.bind.side_effect = [bound_primary, bound_retry]

        prompt = (
            "Header line\n\n"
            + ("Background line with details.\n" * 80)
            + "\n### Day 7: Departure\n- **Morning:** Pack up\n\nInstructions:\n- Keep all 7 days."
        )

        with patch("backend.llm", mock_llm):
            content, is_fallback = backend._safe_llm_invoke(
                system_prompt="System",
                user_prompt=prompt,
                max_tokens=2048,
            )

        self.assertFalse(is_fallback)
        self.assertIn("### Day 1: Recovered", content)
        # Verify retry kept max_tokens=2048 (not slashed to 1024) and preserved newlines & instructions
        self.assertEqual(mock_llm.bind.call_args_list[1].kwargs.get("max_tokens"), 2048)
        retry_human_msg = bound_retry.invoke.call_args.args[0][1].content
        self.assertIn("\n", retry_human_msg)
        self.assertIn("Instructions:\n- Keep all 7 days.", retry_human_msg)
        self.assertIn("### Day 7: Departure", retry_human_msg)

    def test_itinerary_agent_requests_all_days_with_full_budget(self):
        state = {
            "user_query": "Plan a 7-day trip to Japan",
            "trip_constraints": {"destination": "Japan", "duration": "7 days"},
            "structured_trip_state": {"dates": {"duration_days": 7}},
            "flight_results": "ANA direct flight ($900).",
            "hotel_results": "- **Hotel Gracery Shinjuku**: Great stay ([Info](https://example.com/1))",
            "weather_results": "Tokyo: 18C clear.",
            "budget_results": "Budget is feasible.",
            "llm_calls": 2,
        }
        captured = {}

        def fake_invoke(system_prompt, user_prompt, fallback_fn=None, max_tokens=None, retry_user_prompt=None):
            captured["user_prompt"] = user_prompt
            captured["max_tokens"] = max_tokens
            captured["retry_user_prompt"] = retry_user_prompt
            return _make_detailed_itinerary(7, "Japan"), False

        with patch("backend._safe_llm_invoke", side_effect=fake_invoke):
            res = backend.itinerary_agent(state)

        self.assertEqual(captured["max_tokens"], backend.MAX_COMPLETION_TOKENS)
        self.assertIn("Day 1 through ### Day 7", captured["user_prompt"])
        self.assertIn("Day 1 through ### Day 7", captured["retry_user_prompt"])
        _assert_all_days_have_meaningful_activities(self, res["itinerary"], 7)

    def test_final_agent_approved_preserves_long_7_day_itinerary_and_specialists_without_llm(self):
        long_7_day = _make_detailed_itinerary(7, "Japan")
        self.assertGreater(len(long_7_day), 3000)

        flight_info = "ANA NH846 Dhaka to Tokyo Narita ($920 roundtrip). Book 6 weeks in advance."
        hotel_info = (
            "- **Hotel Gracery Shinjuku**: Central Shinjuku stay near JR station ([Info](https://example.com/h1))\n"
            "- **Shiba Park Hotel**: Quiet central Tokyo hotel near Tokyo Tower ([Info](https://example.com/h2))\n"
            "- **Hotel Monterey Kyoto**: Boutique stay in Nakagyo Ward ([Info](https://example.com/h3))"
        )
        weather_info = "Current in Japan: 19°C, clear sky, 52% humidity."
        budget_info = "Estimated Total: $2,100 (Flights $920, Hotels $780, Food/Transit $400). Feasible."

        state = {
            "user_query": "Plan a 7-day Japan trip",
            "trip_constraints": {"destination": "Japan", "duration": "7 days", "budget": "$2200"},
            "flight_results": flight_info,
            "hotel_results": hotel_info,
            "weather_results": weather_info,
            "budget_results": budget_info,
            "itinerary": long_7_day,
            "approved": True,
            "human_feedback": "",
            "llm_calls": 5,
        }

        with patch("backend._safe_llm_invoke") as mock_invoke:
            res = backend.final_agent(state)

        mock_invoke.assert_not_called()
        final_text = res["final_response"]
        self.assertEqual(res["llm_calls"], 5)
        _assert_all_days_have_meaningful_activities(self, final_text, 7)
        self.assertIn(flight_info, final_text)
        self.assertIn(hotel_info, final_text)
        self.assertIn(weather_info, final_text)
        self.assertIn(budget_info, final_text)

    def test_final_agent_revision_passes_complete_itinerary_without_3000_char_truncation(self):
        long_7_day = _make_detailed_itinerary(7, "Japan")
        self.assertGreater(len(long_7_day), 3000)

        flight_info = "ANA direct ($920)."
        hotel_info = "- **Hotel Gracery Shinjuku**: Central Shinjuku ([Info](https://example.com/h1))"
        weather_info = "19°C clear."
        budget_info = "Total ~$2,100."

        state = {
            "user_query": "Plan a 7-day Japan trip",
            "trip_constraints": {"destination": "Japan", "duration": "7 days", "budget": "$2200"},
            "structured_trip_state": {"dates": {"duration_days": 7}},
            "flight_results": flight_info,
            "hotel_results": hotel_info,
            "weather_results": weather_info,
            "budget_results": budget_info,
            "itinerary": long_7_day,
            "approved": False,
            "human_feedback": "Make Day 7 include a morning Tsukiji Outer Market food tour.",
            "llm_calls": 5,
        }

        captured = {}
        revised_7_day = long_7_day.replace(
            "Visit Landmark 7A in Japan",
            "Take a morning Tsukiji Outer Market food tour in Tokyo",
        )

        def fake_invoke(system_prompt, user_prompt, fallback_fn=None, max_tokens=None, retry_user_prompt=None):
            captured["user_prompt"] = user_prompt
            captured["retry_user_prompt"] = retry_user_prompt
            captured["max_tokens"] = max_tokens
            return backend._clean_itinerary_for_final(revised_7_day), False

        with patch("backend._safe_llm_invoke", side_effect=fake_invoke):
            res = backend.final_agent(state)

        # Ensure Day 7 from the >3000-char draft reached the revision prompt intact with newlines
        self.assertIn("### Day 7: Japan Highlights Part 7", captured["user_prompt"])
        self.assertIn("Traditional Kitchen 7E", captured["user_prompt"])
        self.assertIn("\n### Day 4:", captured["user_prompt"])
        self.assertIn("### Day 7: Japan Highlights Part 7", captured["retry_user_prompt"])
        self.assertEqual(captured["max_tokens"], backend.MAX_COMPLETION_TOKENS)

        final_text = res["final_response"]
        _assert_all_days_have_meaningful_activities(self, final_text, 7)
        self.assertIn("Tsukiji Outer Market food tour", final_text)
        self.assertIn(flight_info, final_text)
        self.assertIn(hotel_info, final_text)
        self.assertIn(weather_info, final_text)
        self.assertIn(budget_info, final_text)
        self.assertEqual(res["llm_calls"], 6)


if __name__ == "__main__":
    unittest.main()
