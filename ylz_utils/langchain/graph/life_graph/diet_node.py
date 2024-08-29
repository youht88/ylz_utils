from  .state import Diets,State
from .node import Node

from langchain_core.messages import AIMessage
class DietNode(Node):
    def dietNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Diets)
        message = state["messages"][-1]
        res = llm_with_output.invoke([message])
        if isinstance(res,Diets):
            return {"messages":[AIMessage(content=str(res))]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析饮食数据")]}  