"""Defines the configurable parameters for the agent.
Is a place where the agent can store information about the current context, such as the site_id and products,
which is not part of the system prompt, but is still relevant to the agent's operation."""

from __future__ import annotations
from typing import Any

from pydantic import BaseModel

class Context(BaseModel):
    model: str
    system_prompt: str
    site_id: int
    products: list[dict[str, Any]]
