from langchain_core.messages import ToolMessage,AIMessage

from .state import State,RequestAssistance
from .node import Node

class ChatbotNode(Node):
    def chatbot(self,state:State):
        print("begin invoke....")
        print(state["messages"])
        try:
            # if isinstance(state["messages"][-1],ToolMessage):
            #     return {"messages":[AIMessage(state["messages"][-1].content)]}
            response = self.standGraph.llm_with_tools.invoke(state["messages"])
            #response = self.standGraph.llm.invoke(state["messages"])
            print("end invoke!!!")
            print("\nChatbotNode",response)
            return {"messages":[response]}
        except Exception as e:
           print("ERROR!!!",state)
           raise e