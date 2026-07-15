"""Define the state structures for the agent."""

from typing import Any
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep

class InputState(TypedDict):
    messages: list[AnyMessage]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_products: list[dict[str, Any]]
    is_last_step: IsLastStep