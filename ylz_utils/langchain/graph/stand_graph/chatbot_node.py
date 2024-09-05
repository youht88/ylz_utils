from langchain_core.messages import ToolMessage,AIMessage

from .state import State,RequestAssistance
from .node import Node

class ChatbotNode(Node):
    def __init__(self,graphLib,msg=None):
        super().__init__(graphLib,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_with_tools = self.llm.bind_tools(self.graphLib.tools)
    def __call__(self,state:State):
        print("begin invoke....")
        print(state["messages"])
        try:
            # if isinstance(state["messages"][-1],ToolMessage):
            #     return {"messages":[AIMessage(state["messages"][-1].content)]}
            response = self.llm_with_tools.invoke(state["messages"])
            #response = self.standGraph.llm.invoke(state["messages"])
            print("end invoke!!!")
            print("\nChatbotNode",response)
            return {"messages":[response]}
        except Exception as e:
           print("ERROR!!!",state)
           raise e