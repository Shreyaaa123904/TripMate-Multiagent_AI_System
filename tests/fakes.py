"""Deterministic stand-ins for Groq, Tavily and AviationStack (no secrets needed)."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

ITINERARY = "Day 1: Arrive in Tokyo. Day 2: Kyoto."
FINAL_ANSWER = "## Trip Summary\nA fake 7-day Japan plan."


def install(backend_module):
    llm = FakeListChatModel(responses=[ITINERARY, FINAL_ANSWER])
    backend_module.get_llm = lambda: llm
    backend_module.search_flights = lambda query: "Fake flight results"
    backend_module.tavily_search = lambda query: "Fake hotel results"
    return llm
