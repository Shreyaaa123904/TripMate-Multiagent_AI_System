import logging
import os
import threading

from tavily import TavilyClient

import config  # noqa: F401  (loads .env)

logger = logging.getLogger("tripmate.tools.tavily")

_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _client


def tavily_search(query):
    try:
        response = get_client().search(
            query=query,
            max_results=5
        )
    except Exception as e:
        logger.warning("Tavily search failed: %s", type(e).__name__)
        return "Hotel search is temporarily unavailable."

    results = []

    for i, r in enumerate(response.get("results", []), 1):
        title   = r.get("title", "Unknown")
        url     = r.get("url", "")
        snippet = r.get("content", "").strip()
        # Keep only the first 300 characters to avoid wall-of-text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)
