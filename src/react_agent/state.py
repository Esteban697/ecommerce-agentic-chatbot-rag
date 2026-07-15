"""Defines the state structures for the agent.
State is a TypedDict that represents the current state of the conversation, including the messages exchanged and any retrieved products.
InputState is a TypedDict that represents the input state of the conversation, which includes only the messages exchanged.
IsLastStep is a type that indicates whether the current step is the last step in the conversation.
"""

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