"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context


async def search_web(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

# react_agent/tools.py
from typing import Any
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

def retrieve_products(site_id: int, query: str, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_l = query.lower()
    filtered = [p for p in products if p.get("site_id") == site_id]

    matches = []
    for p in filtered:
        haystack = " ".join(
            str(p.get(k, "")) for k in ["name", "title", "description", "category", "brand"]
        ).lower()
        if any(term in haystack for term in query_l.split()):
            matches.append(p)

    return matches[:5]

#Direct file search tool (MOCK)
async def search_catalog(
    query: str,
    state: Annotated[dict, InjectedState],
) -> Command:
    """Search products in the current catalog and return relevant products."""
    runtime = state["_runtime"]
    site_id = runtime.context.site_id
    products = runtime.context.products

    retrieved = retrieve_products(site_id, query, products)

    summary = []
    for p in retrieved:
        name = p.get("name") or p.get("title")
        price = p.get("price")
        summary.append(f"{name} - price: {price}")

    return Command(
        update={
            "retrieved_products": retrieved,
            "messages": [
                {
                    "role": "tool",
                    "content": "Retrieved products:\n" + "\n".join(summary) if summary else "No products found."
                }
            ],
        }
    )

TOOLS: List[Callable[..., Any]] = [search_web, search_catalog]
