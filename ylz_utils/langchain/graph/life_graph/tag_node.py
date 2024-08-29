from  .state import Tag,State
from .node import Node

from langchain_core.messages import AIMessage

class TagNode(Node):
    def tagNode(self,state:State):
        llm = self.get_llm("LLM.DEEPBRICKS")
        llm_with_output = llm.with_structured_output(Tag)
        message = state["messages"][-1]
        tag = llm_with_output.invoke([message])
        if isinstance(tag,Tag):
            return {"life_tag":tag}
        else:
            return {"human":True}  