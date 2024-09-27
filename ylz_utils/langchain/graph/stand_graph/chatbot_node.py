from langchain_core.messages import ToolMessage,AIMessage,BaseMessage,RemoveMessage,HumanMessage

from .state import State,RequestAssistance
from .node import Node

from rich.console import Console

class ChatbotNode(Node):
    def __init__(self,graphLib,msg=None):
        super().__init__(graphLib,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_with_tools = self.llm.bind_tools(self.graphLib.tools)
    def __call__(self,state:State):
        print("begin invoke....")
        lastMessage:BaseMessage = state["messages"][-1]
        messages = state["messages"]
        if isinstance(lastMessage,ToolMessage):
            if  not lastMessage.content:
               lastMessage.content="我没有从工具中获得信息，如果你使用python_repl工具则必须在最后使用print(...)语句返回计算结果" 
            try:
                response = self.llm_with_tools.invoke(state["messages"])
                response = self.graphLib.get_safe_response(response)
                return {"messages":[response]}
            except Exception as e:
                Console().print("ERROR!!!",messages,e)
                raise e
        else:
            response = self.llm_with_tools.invoke(messages)
            response = self.graphLib.get_safe_response(response)
            Console().print("HERE NO TOOL!!!")
            return {"messages":[response]}
        