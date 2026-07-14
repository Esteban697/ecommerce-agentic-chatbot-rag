"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, TypedDict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

class InputState(TypedDict):
    messages: list

class State(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_products: list[dict[str, Any]]
    is_last_step: bool