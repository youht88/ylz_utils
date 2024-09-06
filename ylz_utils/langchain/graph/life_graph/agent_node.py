from  .state import Diets,State
from .node import Node

from langchain_core.messages import AIMessage
class AgentNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
    def __call__(self,state:State):
        messages = state["messages"]
        print("messages--->",messages)
        res = self.llm.invoke(messages)
        return {"messages":[res]}  