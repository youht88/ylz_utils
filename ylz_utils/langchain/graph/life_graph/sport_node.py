from  .state import Sports,State
from .node import Node

from langchain_core.messages import AIMessage
class SportNode(Node):
    def sportNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Sports)
        message = state["messages"][-1]
        res = llm_with_output.invoke([message])
        if isinstance(res,Sports):
            return {"messages":[AIMessage(content=str(res))]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析运动数据")]}  