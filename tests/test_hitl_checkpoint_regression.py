import os
import unittest
import uuid
from decimal import Decimal
from typing import Any, Annotated, TypedDict
import operator

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from trip_state import (
    StructuredTripState,
    TripLocation,
    LocationGranularity,
    TravellerCounts,
    TripDates,
    TripOperation,
    merge_trip_state,
)
from trip_extraction import TripExtractionResult


load_dotenv()


# ==============================================================================
# Graph State & Test Specialist Nodes
# ==============================================================================
class TestTravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str
    structured_trip_state: dict[str, Any] | None
    structured_trip_provenance: dict[str, str]
    structured_trip_last_operation: str


class HITLRegressionSupervisor:
    def __init__(self):
        self.flight_search_calls = 0
        self.hotel_search_calls = 0
        self.budget_eval_calls = 0
        self.itinerary_draft_calls = 0
        self.final_agent_calls = 0
        self.last_flight_destination = None
        self.last_hotel_destination = None

    def supervisor_node(self, state: TestTravelState) -> dict[str, Any]:
        provenance = state.get("structured_trip_provenance", {})
        constraints = state.get("trip_constraints", {})

        # Route specialists based on provenance of changes
        selected = []
        dest_changed = provenance.get("destination") == "merged" or not state.get("flight_results")
        budget_changed = provenance.get("budget_amount") == "merged" or provenance.get("budget_currency") == "merged"

        if dest_changed:
            selected.extend(["flight_agent", "hotel_agent"])
        if budget_changed or not state.get("budget_results"):
            selected.append("budget_agent")

        selected.append("itinerary_agent")
        return {"selected_agents": selected}

    def flight_agent_node(self, state: TestTravelState) -> dict[str, Any]:
        self.flight_search_calls += 1
        dest = state.get("trip_constraints", {}).get("destination", "Unknown")
        self.last_flight_destination = dest
        return {
            "flight_results": f"Confirmed flights for {dest} (Route: Origin->{dest})",
            "messages": [AIMessage(content=f"Flight fetched for {dest}")],
        }

    def hotel_agent_node(self, state: TestTravelState) -> dict[str, Any]:
        self.hotel_search_calls += 1
        dest = state.get("trip_constraints", {}).get("destination", "Unknown")
        self.last_hotel_destination = dest
        return {
            "hotel_results": f"Selected top hotels in {dest}",
            "messages": [AIMessage(content=f"Hotels fetched for {dest}")],
        }

    def budget_agent_node(self, state: TestTravelState) -> dict[str, Any]:
        self.budget_eval_calls += 1
        budget = state.get("trip_constraints", {}).get("budget", "N/A")
        return {
            "budget_results": f"Budget plan validated: {budget}",
            "messages": [AIMessage(content=f"Budget analyzed: {budget}")],
        }

    def itinerary_agent_node(self, state: TestTravelState) -> dict[str, Any]:
        self.itinerary_draft_calls += 1
        constraints = state.get("trip_constraints", {})
        dest = constraints.get("destination", "")
        duration = constraints.get("duration", "")
        budget = constraints.get("budget", "")
        flights = state.get("flight_results", "")
        hotels = state.get("hotel_results", "")

        draft = (
            f"Draft Itinerary for {dest} ({duration}). "
            f"Budget: {budget}. "
            f"Flights: [{flights}]. "
            f"Hotels: [{hotels}]."
        )
        return {
            "itinerary": draft,
            "approval_request": f"Please review and approve draft itinerary for {dest}.",
            "messages": [AIMessage(content="Draft itinerary generated.")],
        }

    def human_approval_node(self, state: TestTravelState) -> dict[str, Any]:
        # HITL node interrupts execution with draft
        review = interrupt(
            {
                "draft_itinerary": state.get("itinerary", ""),
                "approval_request": state.get("approval_request", ""),
            }
        )
        approved = bool(review.get("approved", False))
        feedback = str(review.get("feedback", "")).strip()
        return {
            "approved": approved,
            "human_feedback": feedback,
            "messages": [AIMessage(content=f"HITL completed (approved={approved}).")],
        }

    def final_agent_node(self, state: TestTravelState) -> dict[str, Any]:
        self.final_agent_calls += 1
        approved = state.get("approved", False)
        feedback = state.get("human_feedback", "")
        draft = state.get("itinerary", "")
        final_doc = f"FINAL ITINERARY PLAN: Based on '{draft}'. Approved={approved}. Feedback='{feedback}'."
        return {
            "final_response": final_doc,
            "messages": [AIMessage(content=final_doc)],
        }


def _build_test_graph(supervisor: HITLRegressionSupervisor):
    g = StateGraph(TestTravelState)
    g.add_node("supervisor", supervisor.supervisor_node)
    g.add_node("flight_agent", supervisor.flight_agent_node)
    g.add_node("hotel_agent", supervisor.hotel_agent_node)
    g.add_node("budget_agent", supervisor.budget_agent_node)
    g.add_node("itinerary_agent", supervisor.itinerary_agent_node)
    g.add_node("human_approval", supervisor.human_approval_node)
    g.add_node("final_agent", supervisor.final_agent_node)

    route_map = {
        "flight_agent": "flight_agent",
        "hotel_agent": "hotel_agent",
        "budget_agent": "budget_agent",
        "itinerary_agent": "itinerary_agent",
    }
    agent_order = ["flight_agent", "hotel_agent", "budget_agent", "itinerary_agent"]

    def route_supervisor(state: TestTravelState) -> str:
        selected = state.get("selected_agents", [])
        for agent in agent_order:
            if agent in selected:
                return agent
        return "itinerary_agent"

    def route_after(current: str):
        def _route(state: TestTravelState) -> str:
            selected = state.get("selected_agents", [])
            idx = agent_order.index(current)
            for nxt in agent_order[idx + 1 :]:
                if nxt in selected:
                    return nxt
            return "itinerary_agent"
        return _route

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_supervisor, route_map)
    g.add_conditional_edges("flight_agent", route_after("flight_agent"), route_map)
    g.add_conditional_edges("hotel_agent", route_after("hotel_agent"), route_map)
    g.add_conditional_edges("budget_agent", route_after("budget_agent"), route_map)
    g.add_edge("itinerary_agent", "human_approval")
    g.add_edge("human_approval", "final_agent")
    g.add_edge("final_agent", END)

    return g


def _build_provenance(saved: StructuredTripState | None, merged: StructuredTripState | None, incoming: StructuredTripState | None) -> dict[str, str]:
    fields = ["origin", "destination", "dates", "travellers", "budget_amount", "budget_currency"]
    prov = {}
    for f in fields:
        s_val = getattr(saved, f, None) if saved else None
        m_val = getattr(merged, f, None) if merged else None
        i_val = getattr(incoming, f, None) if incoming else None
        if i_val is not None and m_val != s_val:
            prov[f] = "merged"
        else:
            prov[f] = "saved"
    return prov


def _constraints_from_structured(structured: StructuredTripState | None) -> dict[str, Any]:
    c = {"destination": "", "origin": "", "duration": "", "budget": ""}
    if not structured:
        return c
    if structured.destination:
        c["destination"] = structured.destination.city or structured.destination.country or structured.destination.raw_text or ""
    if structured.origin:
        c["origin"] = structured.origin.city or structured.origin.country or structured.origin.raw_text or ""
    if structured.dates and structured.dates.duration_days:
        c["duration"] = f"{structured.dates.duration_days} days"
    if structured.budget_amount is not None:
        curr = structured.budget_currency or "$"
        c["budget"] = f"{curr} {structured.budget_amount}"
    return c


# ==============================================================================
# Isolated Regression Test Suite
# ==============================================================================
class HITLCheckpointRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_conn = None
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            try:
                if "sslmode=" not in db_url:
                    db_url += "?sslmode=require"
                conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row, connect_timeout=5)
                saver = PostgresSaver(conn)
                saver.setup()
                cls.db_conn = conn
                cls.checkpointer = saver
                cls.backend_type = "PostgreSQL (Render Remote)"
            except Exception as exc:
                print(f"PostgreSQL unavailable ({exc}), testing with InMemorySaver")
                cls.checkpointer = InMemorySaver()
                cls.backend_type = "InMemorySaver (Fallback)"
        else:
            cls.checkpointer = InMemorySaver()
            cls.backend_type = "InMemorySaver (Fallback)"

    @classmethod
    def tearDownClass(cls):
        if cls.db_conn:
            try:
                cls.db_conn.close()
            except Exception:
                pass

    def setUp(self):
        self.supervisor = HITLRegressionSupervisor()
        self.state_graph = _build_test_graph(self.supervisor)
        self.app = self.state_graph.compile(checkpointer=self.checkpointer)

    def _execute_turn(self, thread_id: str, message: str, incoming_state: StructuredTripState) -> tuple[dict[str, Any], Any]:
        config = {"configurable": {"thread_id": thread_id}}

        # Load saved structured state from checkpoint
        snapshot = self.app.get_state(config)
        saved_raw = getattr(snapshot, "values", {}).get("structured_trip_state")
        saved_state = StructuredTripState.model_validate(saved_raw) if saved_raw else None

        # Merge state
        outcome = merge_trip_state(saved_state, incoming_state)
        merged_state = outcome.merged_state
        provenance = _build_provenance(saved_state, merged_state, incoming_state)
        constraints = _constraints_from_structured(merged_state)

        # Context preservation logic:
        # If destination changed, invalidate stale destination-dependent results.
        # If destination has not changed, do NOT wipe out existing flight/hotel results.
        dest_changed = provenance.get("destination") == "merged"

        payload: dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "user_query": message,
            "trip_constraints": constraints,
            "structured_trip_state": merged_state.model_dump(mode="json"),
            "structured_trip_provenance": provenance,
            "structured_trip_last_operation": incoming_state.operation.value,
        }

        if dest_changed and saved_state and saved_state.destination:
            # Explicitly clear stale results for the new destination
            payload["flight_results"] = ""
            payload["hotel_results"] = ""

        result = self.app.invoke(payload, config=config)
        return result, snapshot

    def test_complete_hitl_multi_turn_lifecycle_and_invalidation(self):
        """
        Verifies:
        1. Turn 1 creates a draft and pauses at human_approval.
        2. Turn 2 arrives using the same thread_id while the first approval is pending.
        3. The new message updates the existing trip context and generates an updated draft.
        4. The latest approval interrupt can be resumed successfully.
        5. The old draft cannot accidentally be approved.
        6. A budget-only update preserves existing flight and hotel results.
        7. A destination change invalidates stale destination-dependent results and triggers fresh searches.
        """
        print(f"\n[Test Environment] Running against checkpointer: {self.backend_type}")
        thread_id = f"hitl_regress_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}

        # ----------------------------------------------------------------------
        # 1. Turn 1: "Plan a 7-day Nepal trip for 2 adults."
        # ----------------------------------------------------------------------
        t1_input = StructuredTripState(
            operation=TripOperation.NEW_TRIP,
            destination=TripLocation(granularity=LocationGranularity.COUNTRY, country="Nepal"),
            dates=TripDates(duration_days=7),
            travellers=TravellerCounts(adults=2),
        )
        r1, _ = self._execute_turn(thread_id, "Plan a 7-day Nepal trip for 2 adults.", t1_input)

        # 1.1 Assert paused at human_approval
        snapshot_t1 = self.app.get_state(config)
        self.assertEqual(snapshot_t1.next, ("human_approval",), "Turn 1 must pause at human_approval")
        self.assertIn("__interrupt__", r1, "Turn 1 must return an active interrupt")
        t1_interrupt = r1["__interrupt__"][0]
        t1_interrupt_id = getattr(t1_interrupt, "id", None) or getattr(snapshot_t1.tasks[0].interrupts[0], "id")
        t1_draft = t1_interrupt.value["draft_itinerary"]

        # 1.2 Assert draft content and tool calls
        self.assertIn("Nepal", t1_draft)
        self.assertIn("7 days", t1_draft)
        self.assertEqual(self.supervisor.flight_search_calls, 1, "Flight search must run on initial destination")
        self.assertEqual(self.supervisor.hotel_search_calls, 1, "Hotel search must run on initial destination")
        self.assertEqual(self.supervisor.last_flight_destination, "Nepal")
        self.assertEqual(self.supervisor.last_hotel_destination, "Nepal")

        # ----------------------------------------------------------------------
        # 2 & 3 & 6. Turn 2: "My budget is $2,000." (Budget-only update on same thread_id while Turn 1 approval is pending)
        # ----------------------------------------------------------------------
        t2_input = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_amount=Decimal("2000"),
            budget_currency="USD",
        )
        r2, _ = self._execute_turn(thread_id, "My budget is $2,000.", t2_input)

        # 2. Assert same thread_id accepted while pending
        snapshot_t2 = self.app.get_state(config)
        self.assertEqual(snapshot_t2.next, ("human_approval",), "Turn 2 must pause at human_approval")

        # 3. Assert draft updated with new budget and preserved details
        t2_interrupt = r2["__interrupt__"][0]
        t2_interrupt_id = getattr(t2_interrupt, "id", None) or getattr(snapshot_t2.tasks[0].interrupts[0], "id")
        t2_draft = t2_interrupt.value["draft_itinerary"]
        self.assertIn("Nepal", t2_draft, "Nepal destination must be preserved in Turn 2 draft")
        self.assertIn("7 days", t2_draft, "7 days duration must be preserved in Turn 2 draft")
        self.assertIn("USD 2000", t2_draft, "USD 2000 budget must appear in Turn 2 draft")

        # 6. Assert budget-only update preserved flight and hotel results without re-calling tools
        self.assertEqual(self.supervisor.flight_search_calls, 1, "Flight search must NOT re-run on budget-only update")
        self.assertEqual(self.supervisor.hotel_search_calls, 1, "Hotel search must NOT re-run on budget-only update")
        self.assertIn("Confirmed flights for Nepal", t2_draft, "Turn 1 flight results must be preserved in Turn 2 draft")
        self.assertIn("Selected top hotels in Nepal", t2_draft, "Turn 1 hotel results must be preserved in Turn 2 draft")

        # Verify interrupt was refreshed (new interrupt ID replaces old one)
        self.assertNotEqual(t1_interrupt_id, t2_interrupt_id, "Turn 2 must establish a fresh interrupt ID")

        # ----------------------------------------------------------------------
        # 7. Turn 3: "Change destination to Thailand." (Destination change on same thread_id)
        # ----------------------------------------------------------------------
        t3_input = StructuredTripState(
            operation=TripOperation.UPDATE,
            destination=TripLocation(granularity=LocationGranularity.COUNTRY, country="Thailand"),
        )
        r3, _ = self._execute_turn(thread_id, "Change destination to Thailand.", t3_input)

        snapshot_t3 = self.app.get_state(config)
        self.assertEqual(snapshot_t3.next, ("human_approval",), "Turn 3 must pause at human_approval")
        t3_interrupt = r3["__interrupt__"][0]
        t3_draft = t3_interrupt.value["draft_itinerary"]

        # 7.1 Assert destination change triggered fresh searches
        self.assertEqual(self.supervisor.flight_search_calls, 2, "Flight search MUST re-run when destination changes")
        self.assertEqual(self.supervisor.hotel_search_calls, 2, "Hotel search MUST re-run when destination changes")
        self.assertEqual(self.supervisor.last_flight_destination, "Thailand")
        self.assertEqual(self.supervisor.last_hotel_destination, "Thailand")

        # 7.2 Assert stale results were invalidated and replaced with Thailand results
        self.assertIn("Thailand", t3_draft, "Turn 3 draft must be for Thailand")
        self.assertNotIn("Nepal", t3_draft, "Stale Nepal flight/hotel/dest must not appear in Turn 3 draft")
        self.assertIn("Confirmed flights for Thailand", t3_draft)
        self.assertIn("Selected top hotels in Thailand", t3_draft)
        self.assertIn("USD 2000", t3_draft, "Budget from Turn 2 must be preserved across destination change")

        # ----------------------------------------------------------------------
        # 4 & 5. Resume and verify old draft cannot be approved
        # ----------------------------------------------------------------------
        # The user reviews the latest Thailand draft and clicks "Approve"
        resume_cmd = Command(resume={"approved": True, "feedback": "Looks fantastic!"})
        final_result = self.app.invoke(resume_cmd, config=config)

        # 4. Assert latest approval interrupt resumed successfully
        snapshot_final = self.app.get_state(config)
        self.assertEqual(snapshot_final.next, (), "Resumed thread must complete to END")
        self.assertEqual(self.supervisor.final_agent_calls, 1, "Final agent must execute on resume")

        # 5. Assert old draft cannot accidentally be approved
        final_response = final_result.get("final_response", "")
        self.assertIn("Thailand", final_response, "Approved final itinerary must be for Thailand")
        self.assertNotIn("Nepal", final_response, "Old Nepal draft must NOT be approved or in final plan")
        self.assertIn("USD 2000", final_response, "Final response must reflect accumulated budget")
        self.assertIn("Looks fantastic!", final_response, "Human feedback must be incorporated")

        # Verify thread cannot be resumed again (no pending interrupt remains)
        with self.assertRaises(Exception):
            self.app.invoke(Command(resume={"approved": True}), config=config)


if __name__ == "__main__":
    unittest.main()
