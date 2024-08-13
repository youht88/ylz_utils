from langgraph.graph.state import CompiledStateGraph

from operator import itemgetter
from typing import Literal,List,Annotated
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import TypedDict
from langchain_core.tools import tool,Tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.state import CompiledStateGraph

from ylz_utils.langchain.graph.common import State,RequestAssistance

class StandGraph():
    def __init__(self,graphLib):
        self.graphLib = graphLib

    def chatbot(self,state:State):
        response = self.llm_with_tools.invoke(state["messages"])
        ask_human = False
        if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
            ask_human = True
        return {"messages":[response],"ask_human":ask_human}
    
    def router(self,state:MessagesState)-> Literal["tools","__end__"]:
        messages = state["messages"]
        tool_calls = messages[-1].tool_calls
        if tool_calls:
            return "tools"
        else:
            return END
    def get_graph(self,llm_key=None,llm_model=None) -> CompiledStateGraph:
        llm = self.graphLib.langchainLib.get_llm(llm_key,llm_model)
        tools = [
            self.graphLib.python_repl_tool,
        ]
        if self.graphLib.websearch_tool:
            tools.append(self.graphLib.websearch_tool)
        if self.graphLib.ragsearch_tool:
            tools.append(self.graphLib.ragsearch_tool)
        # tools = [
        #        Tool(name="websearch",
        #             description = "the tool when you can not sure,please search from the internet",
        #             func = websearch
        #        ),
        #        Tool(name="python_repl",
        #             description="when you need to calculation, use python repl tool to execute code ,then return the result to AI.",
        #             func=self.python_repl_tool)
        #      ]
        self.llm_with_tools = llm.bind_tools(tools+[ RequestAssistance ])      

        workflow = StateGraph(State)
        workflow.add_node("translate",self.graphLib.translate_to_en)
        workflow.add_node("chatbot",self.chatbot)
        workflow.add_node("tools",ToolNode(tools=tools))
        workflow.add_node("human", self.graphLib.human_node)
        
        workflow.add_conditional_edges("chatbot", self.graphLib.select_next_node,
                                       {"human":"human","tools":"tools","__end__":"__end__"})
        
        workflow.add_edge(START, "translate")
        workflow.add_edge("translate","chatbot")
        workflow.add_edge("tools","chatbot")
        workflow.add_edge("human","chatbot")

        graph = workflow.compile(checkpointer=self.graphLib.memory,
                                 interrupt_before= ["human"])

        return graph    
