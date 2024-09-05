from  .state import Tag,State
from .node import Node

from langchain_core.messages import AIMessage

class TagNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_with_output = self.llm.with_structured_output(Tag)
    def __call__(self,state:State):
        messages= state["messages"]
        message = messages[-1]
        prompt = self.graphLib.langchainLib.get_prompt()
        #tag = (prompt | self.llm_with_output).invoke({"input":message.content})
        tag = self.llm_with_output.invoke(messages)
        if isinstance(tag,Tag):
            return {"life_tag":tag}
        else:
            return {"human":True}  