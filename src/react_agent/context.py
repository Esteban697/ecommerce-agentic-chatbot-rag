"""Define the configurable parameters for the agent."""

#from __future__ import annotations
from typing import Any

from pydantic import BaseModel

class Context(BaseModel):
    model: str
    system_prompt: str
    site_id: int
    products: list[dict[str, Any]]
