"""Tools for the React agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolRuntime

from react_agent.context import Context


def retrieve_products(
    site_id: int,
    query: str,
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a small list of products relevant to the query for the given site."""
    query_l = query.lower()
    filtered = [p for p in products if p.get("site_id") == site_id]

    matches: list[dict[str, Any]] = []
    for p in filtered:
        haystack = " ".join(
            str(p.get(k, "")) for k in ["name", "title", "description", "category", "brand"]
        ).lower()

        if any(term in haystack for term in query_l.split()):
            matches.append(p)

    return matches[:5]


@tool
async def search_web(query: str, runtime: ToolRuntime[Context]) -> dict[str, Any]:
    """Search the web for up-to-date information."""
    max_results = getattr(runtime.context, "max_search_results", 5)
    wrapped = TavilySearch(max_results=max_results)
    result = await wrapped.ainvoke({"query": query})
    return dict(result)


@tool
async def search_catalog(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search products in the current site catalog and return a concise summary."""
    site_id = runtime.context.site_id
    products = runtime.context.products

    retrieved = retrieve_products(site_id, query, products)

    if not retrieved:
        return "No products found."

    summary_lines = []
    for p in retrieved:
        name = p.get("name") or p.get("title") or "Unnamed product"
        price = p.get("price", "N/A")
        brand = p.get("brand", "Unknown brand")
        category = p.get("category", "Unknown category")
        summary_lines.append(
            f"- {name} | brand: {brand} | category: {category} | price: {price}"
        )

    return "Retrieved products:\n" + "\n".join(summary_lines)


TOOLS = [search_web, search_catalog]