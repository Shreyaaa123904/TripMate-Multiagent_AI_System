import unittest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

import backend
from trip_extraction import TripExtractionResult
from trip_state import (
    LocationGranularity,
    StructuredTripState,
    TravellerCounts,
    TripDates,
    TripLocation,
    TripOperation,
)


class _MockSupervisorLLM:
    """Mock LLM for Supervisor, Guardrails, and specialist agents."""
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        content = messages[-1].content if messages else ""
        if "input guardrail" in str(messages[0].content).lower() or "guardrail" in content.lower():
            return AIMessage(content='{"allowed": true, "reason": ""}')
        if "supervisor" in str(messages[0].content).lower() or "supervisor" in content.lower():
            return AIMessage(
                content='{"selected_agents": ["itinerary_agent"], "trip_constraints": {}, "reasoning": "Standard plan"}'
            )
        if "draft" in content.lower() or "itinerary" in content.lower():
            return AIMessage(content="Draft itinerary: Day 1 Arrival, Day 2 Sightseeing.")
        return AIMessage(content="Final response: 1. Trip Summary\nTrip details.")

    def bind(self, **_kwargs):
        return self


async def _fake_aviation(*_args, **_kwargs):
    return [{"airport_name": "Test Airport", "iata_code": "TST"}]


async def _fake_tavily(*_args, **_kwargs):
    return {"results": [{"title": "Hotel One", "content": "Nice hotel."}]}


async def _fake_weather(*_args, **_kwargs):
    return {"city": "TestCity", "temperature_c": 22}


async def _fake_forecast(*_args, **_kwargs):
    return {"forecast": []}


class TripStateIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.mock_llm = _MockSupervisorLLM()
        self.p_llm = patch.object(backend, "llm", self.mock_llm)
        self.p_get_llm = patch.object(backend, "get_llm", lambda: self.mock_llm)
        self.p_aviation = patch("backend.aviation_mcp_call", side_effect=_fake_aviation)
        self.p_tavily = patch("backend.tavily_mcp_search", side_effect=_fake_tavily)
        self.p_weather = patch("backend.weather_mcp_search", side_effect=_fake_weather)
        self.p_forecast = patch("backend.forecast_mcp_search", side_effect=_fake_forecast)

        self.p_llm.start()
        self.p_get_llm.start()
        self.p_aviation.start()
        self.p_tavily.start()
        self.p_weather.start()
        self.p_forecast.start()

        self.memory_saver = InMemorySaver()
        self.compiled_graph = backend.graph.compile(checkpointer=self.memory_saver)
        backend._travel_graph = self.compiled_graph

    def tearDown(self):
        backend._travel_graph = None
        self.p_forecast.stop()
        self.p_weather.stop()
        self.p_tavily.stop()
        self.p_aviation.stop()
        self.p_get_llm.stop()
        self.p_llm.stop()

    def _state_values(self, thread_id: str):
        graph = backend.get_graph()
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values

    def test_expected_four_turn_conversation_preserves_unrelated_details(self):
        """
        Tests the user's expected conversation:
        1. 'Plan a 7-day Nepal trip for 2 adults.'
        2. 'My budget is $2,000.'
        3. 'Change destination to Thailand.'
        4. 'Make it 5 days.'
        TripMate must preserve all unrelated trip details when applying each update.
        """
        thread_id = "test_expected_conv_thread"

        # Turn 1
        extraction_1 = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.COUNTRY, country="Nepal"),
                travellers=TravellerCounts(adults=2),
                dates=TripDates(duration_days=7),
            )
        )
        with patch.object(backend, "extract_structured_trip", return_value=extraction_1):
            r1 = backend.run_travel_agent("Plan a 7-day Nepal trip for 2 adults.", thread_id=thread_id)
            self.assertTrue(r1["requires_approval"])
            s1 = StructuredTripState.model_validate(r1["structured_trip_state"])
            self.assertEqual(s1.destination.country, "Nepal")
            self.assertEqual(s1.travellers.adults, 2)
            self.assertEqual(s1.dates.duration_days, 7)
            self.assertIsNone(s1.budget_amount)

        # Turn 2
        extraction_2 = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.UPDATE,
                budget_amount=Decimal("2000"),
                budget_currency="USD",
            )
        )
        with patch.object(backend, "extract_structured_trip", return_value=extraction_2):
            r2 = backend.run_travel_agent("My budget is $2,000.", thread_id=thread_id)
            s2 = StructuredTripState.model_validate(r2["structured_trip_state"])
            # Budget added
            self.assertEqual(s2.budget_amount, Decimal("2000"))
            self.assertEqual(s2.budget_currency, "USD")
            # Unrelated details preserved
            self.assertEqual(s2.destination.country, "Nepal")
            self.assertEqual(s2.travellers.adults, 2)
            self.assertEqual(s2.dates.duration_days, 7)

        # Turn 3
        extraction_3 = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.UPDATE,
                destination=TripLocation(granularity=LocationGranularity.COUNTRY, country="Thailand"),
            )
        )
        with patch.object(backend, "extract_structured_trip", return_value=extraction_3):
            r3 = backend.run_travel_agent("Change destination to Thailand.", thread_id=thread_id)
            s3 = StructuredTripState.model_validate(r3["structured_trip_state"])
            # Destination updated
            self.assertEqual(s3.destination.country, "Thailand")
            # Unrelated details preserved
            self.assertEqual(s3.budget_amount, Decimal("2000"))
            self.assertEqual(s3.budget_currency, "USD")
            self.assertEqual(s3.travellers.adults, 2)
            self.assertEqual(s3.dates.duration_days, 7)

        # Turn 4
        extraction_4 = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.UPDATE,
                dates=TripDates(duration_days=5),
            )
        )
        with patch.object(backend, "extract_structured_trip", return_value=extraction_4):
            r4 = backend.run_travel_agent("Make it 5 days.", thread_id=thread_id)
            s4 = StructuredTripState.model_validate(r4["structured_trip_state"])
            # Duration updated
            self.assertEqual(s4.dates.duration_days, 5)
            # All other details preserved
            self.assertEqual(s4.destination.country, "Thailand")
            self.assertEqual(s4.budget_amount, Decimal("2000"))
            self.assertEqual(s4.budget_currency, "USD")
            self.assertEqual(s4.travellers.adults, 2)

        # Verify state in checkpointer
        checkpoint_values = self._state_values(thread_id)
        saved_final = StructuredTripState.model_validate(checkpoint_values["structured_trip_state"])
        self.assertEqual(saved_final.destination.country, "Thailand")
        self.assertEqual(saved_final.dates.duration_days, 5)
        self.assertEqual(saved_final.budget_amount, Decimal("2000"))
        self.assertEqual(saved_final.travellers.adults, 2)

    def test_ambiguous_fields_do_not_overwrite_valid_saved_state(self):
        thread_id = "test_ambiguous_thread"

        initial = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Paris"),
                budget_amount=Decimal("1500"),
                budget_currency="EUR",
            )
        )
        ambiguous = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.UPDATE,
                destination=TripLocation(granularity=LocationGranularity.UNKNOWN, raw_text="somewhere"),
                budget_currency_ambiguous=True,
                budget_currency_raw="dollars",
            )
        )

        with patch.object(backend, "extract_structured_trip", side_effect=[initial, ambiguous]):
            backend.run_travel_agent("Plan Paris trip with 1500 EUR", thread_id=thread_id)
            r2 = backend.run_travel_agent("Change destination to somewhere maybe 2000 dollars", thread_id=thread_id)

        structured = StructuredTripState.model_validate(r2["structured_trip_state"])
        self.assertEqual(structured.destination.city, "Paris")
        self.assertEqual(structured.budget_currency, "EUR")
        self.assertTrue(structured.budget_currency_ambiguous)
        self.assertTrue(any("Destination is ambiguous" in c for c in r2["structured_trip_clarifications"]))

    def test_follow_up_question_does_not_mutate_state(self):
        thread_id = "test_follow_up_thread"

        initial = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Tokyo"),
                budget_amount=Decimal("3000"),
                budget_currency="USD",
            )
        )
        follow_up = TripExtractionResult(
            state=StructuredTripState(operation=TripOperation.FOLLOW_UP)
        )

        with patch.object(backend, "extract_structured_trip", side_effect=[initial, follow_up]):
            backend.run_travel_agent("Plan Tokyo trip", thread_id=thread_id)
            r2 = backend.run_travel_agent("Is it safe to visit in August?", thread_id=thread_id)

        values = self._state_values(thread_id)
        structured = StructuredTripState.model_validate(values["structured_trip_state"])
        self.assertEqual(structured.destination.city, "Tokyo")
        self.assertEqual(structured.budget_amount, Decimal("3000"))
        self.assertEqual(values.get("structured_trip_last_operation"), "follow_up_question")

    def test_remove_and_reset_operations(self):
        thread_id = "test_remove_reset_thread"

        initial = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Rome"),
                budget_amount=Decimal("1200"),
                budget_currency="EUR",
                travellers=TravellerCounts(adults=2),
            )
        )
        remove_budget = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.REMOVE,
                remove_fields=["budget"],
            )
        )
        reset_trip = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.RESET,
                reset_scope="trip",
            )
        )

        with patch.object(backend, "extract_structured_trip", side_effect=[initial, remove_budget, reset_trip]):
            backend.run_travel_agent("Plan Rome with 1200 EUR", thread_id=thread_id)

            r_removed = backend.run_travel_agent("Remove budget", thread_id=thread_id)
            s_rem = StructuredTripState.model_validate(r_removed["structured_trip_state"])
            self.assertEqual(s_rem.destination.city, "Rome")
            self.assertIsNone(s_rem.budget_amount)
            self.assertIsNone(s_rem.budget_currency)
            self.assertEqual(s_rem.travellers.adults, 2)

            r_reset = backend.run_travel_agent("Reset trip", thread_id=thread_id)
            s_reset = StructuredTripState.model_validate(r_reset["structured_trip_state"])
            self.assertIsNone(s_reset.destination)
            self.assertIsNone(s_reset.budget_amount)
            self.assertIsNone(s_reset.travellers)

    def test_extraction_failure_preserves_valid_saved_state(self):
        thread_id = "test_extract_fail_thread"

        initial = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Bangkok"),
            )
        )
        failed = TripExtractionResult(
            state=None,
            clarifications=["Unable to parse message."],
            extraction_error="validation_error",
        )

        with patch.object(backend, "extract_structured_trip", side_effect=[initial, failed]):
            backend.run_travel_agent("Plan Bangkok", thread_id=thread_id)
            r2 = backend.run_travel_agent("???", thread_id=thread_id)

        values = self._state_values(thread_id)
        structured = StructuredTripState.model_validate(values["structured_trip_state"])
        self.assertEqual(structured.destination.city, "Bangkok")
        self.assertEqual(r2["structured_trip_extraction_error"], "validation_error")
        self.assertIn("Unable to parse message.", r2["structured_trip_clarifications"])

    def test_field_provenance_tracking(self):
        thread_id = "test_provenance_thread"

        initial = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Kyoto"),
                travellers=TravellerCounts(adults=2),
            )
        )
        update = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.UPDATE,
                budget_amount=Decimal("1500"),
                budget_currency="USD",
            )
        )

        with patch.object(backend, "extract_structured_trip", side_effect=[initial, update]):
            backend.run_travel_agent("Plan Kyoto for 2", thread_id=thread_id)
            r2 = backend.run_travel_agent("Budget is $1500", thread_id=thread_id)

        prov = r2["structured_trip_provenance"]
        self.assertEqual(prov.get("budget_amount"), "merged")
        self.assertEqual(prov.get("budget_currency"), "merged")
        self.assertEqual(prov.get("destination"), "saved")
        self.assertEqual(prov.get("travellers"), "saved")

    def test_support_older_checkpoints_without_structured_state(self):
        thread_id = "test_legacy_thread"
        config = {"configurable": {"thread_id": thread_id}}

        # Simulate older checkpoint with no structured_trip_state
        self.compiled_graph.update_state(
            config,
            {
                "user_query": "Old trip",
                "itinerary": "Old itinerary",
            },
        )

        loaded = backend._load_saved_structured_trip_state(backend.get_graph(), config)
        self.assertIsNone(loaded)

        # Now send a new message on the legacy thread
        new_extraction = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Berlin"),
            )
        )
        with patch.object(backend, "extract_structured_trip", return_value=new_extraction):
            r = backend.run_travel_agent("Plan Berlin trip", thread_id=thread_id)

        structured = StructuredTripState.model_validate(r["structured_trip_state"])
        self.assertEqual(structured.destination.city, "Berlin")

    def test_resume_travel_agent_preserves_structured_state(self):
        thread_id = "test_resume_thread"

        initial = TripExtractionResult(
            state=StructuredTripState(
                operation=TripOperation.NEW_TRIP,
                destination=TripLocation(granularity=LocationGranularity.CITY, city="Madrid"),
                travellers=TravellerCounts(adults=2),
                budget_amount=Decimal("1800"),
                budget_currency="EUR",
            )
        )

        with patch.object(backend, "extract_structured_trip", return_value=initial):
            r1 = backend.run_travel_agent("Plan Madrid trip", thread_id=thread_id)
            self.assertTrue(r1["requires_approval"])

        # Resume thread after user approval
        r2 = backend.resume_travel_agent(thread_id=thread_id, approved=True, feedback="Looks great")
        self.assertFalse(r2["requires_approval"])
        self.assertEqual(r2["approved"], True)

        # Verify structured state remains intact after resume
        values = self._state_values(thread_id)
        structured = StructuredTripState.model_validate(values["structured_trip_state"])
        self.assertEqual(structured.destination.city, "Madrid")
        self.assertEqual(structured.travellers.adults, 2)
        self.assertEqual(structured.budget_amount, Decimal("1800"))
        self.assertEqual(structured.budget_currency, "EUR")




if __name__ == "__main__":
    unittest.main()
