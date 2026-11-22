from langgraph.checkpoint.memory import InMemorySaver

import backend
from tests import fakes


def _compile(monkeypatch):
    llm_calls = []
    real = fakes.FakeListChatModel(responses=[fakes.ITINERARY, fakes.FINAL_ANSWER])

    class CountingLLM:
        def invoke(self, messages):
            llm_calls.append(messages)
            return real.invoke(messages)

    monkeypatch.setattr(backend, "get_llm", lambda: CountingLLM())
    monkeypatch.setattr(backend, "search_flights", lambda q: "Fake flight results")
    monkeypatch.setattr(backend, "tavily_search", lambda q: "Fake hotel results")
    graph = backend.graph.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(backend, "_travel_graph", graph)
    return graph, llm_calls


def test_llm_calls_counts_only_real_llm_invocations(monkeypatch):
    _, llm_calls = _compile(monkeypatch)
    result = backend.run_travel_agent("Plan a 7 days Japan trip")
    assert result["llm_calls"] == 2
    assert len(llm_calls) == 2


def test_run_travel_agent_return_shape_unchanged(monkeypatch):
    _compile(monkeypatch)
    result = backend.run_travel_agent("Plan a 7 days Japan trip", thread_id="test_user")
    assert set(result) == {"thread_id", "answer", "flight_results", "hotel_results", "itinerary", "llm_calls"}
    assert result["thread_id"] == "test_user"
    assert result["answer"] == fakes.FINAL_ANSWER
    assert result["itinerary"] == fakes.ITINERARY
    assert result["flight_results"] == "Fake flight results"
    assert result["hotel_results"] == "Fake hotel results"


def test_node_order_unchanged(monkeypatch):
    graph, _ = _compile(monkeypatch)
    order = [list(step)[0] for step in graph.stream(
        {"messages": [], "user_query": "x", "flight_results": "", "hotel_results": "",
         "itinerary": "", "llm_calls": 0},
        config={"configurable": {"thread_id": "t"}},
    )]
    assert order == ["flight_agent", "hotel_agent", "itinerary_agent", "final_agent"]


def test_generated_thread_ids_are_unguessable(monkeypatch):
    _compile(monkeypatch)
    tid = backend.run_travel_agent("hi")["thread_id"]
    assert tid.startswith("user_") and len(tid) == 37
