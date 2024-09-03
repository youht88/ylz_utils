from langchain_core.messages import ToolMessage

from .state import State
from .node import Node

class HumanNode(Node):
    def human_node(state: State):
            new_messages = []
            if not isinstance(state["messages"][-1],ToolMessage):
                new_messages.append(
                    ToolMessage(
                        content = "No response from human.",
                        tool_call_id = state["messages"][-1].tools_calls[0]["id"]
                    )
                )
            return {
                "messages": new_messages,
                "ask_human": False
            }
    