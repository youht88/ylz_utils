from typing import Literal

from langgraph.graph import END,MessagesState
from langgraph.prebuilt import ToolNode
from .state import State

def router(state:State)-> Literal["tools","__end__"]:
    messages = state["messages"]
    tool_calls = messages[-1].tool_calls
    if tool_calls:
        return "tools"
    else:
        return END

