from  .state import Diets,State
from .node import Node

from langchain_core.messages import AIMessage
class AgentNode(Node):
    def agentNode(self,state:State):
        llm = self.get_llm()
        messages = state["messages"]
        res = llm.invoke(messages)
        return {"messages":[res]}  