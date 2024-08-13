from __future__ import annotations
from typing import TYPE_CHECKING,Literal,TypedDict,Annotated
if TYPE_CHECKING:
    from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage


from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from langgraph.graph.state import CompiledStateGraph

class State(TypedDict):
    messages: Annotated[list,add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str   

class StandGraph():
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib
        self.llm_with_tools = None

    def chatbot(self,state:State):
        response = self.llm_with_tools.invoke(state["messages"])
        ask_human = False
        if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
            ask_human = True
        return {"messages":[response],"ask_human":ask_human}
    
    def human_node(self, state: State):
        new_messages = []
        if not isinstance(state["messages"][-1],ToolMessage):
            new_messages.append(
                self.create_response("No response from human.",state["messages"][-1])
            )
        return {
            "messages": new_messages,
            "ask_human": False
        }
    
    def select_next_node(self, state:State) -> Literal["human","tools","__end__"]:
        if state["ask_human"]:
            return "human"
        return tools_condition(state)

    def translate_to_en(self,state:State):
        return state
        content = state["messages"][-1].content
        prompt = self.langchainLib.get_prompt(
"""
将以下句子翻译成英文,注意不要遗漏任何信息，数字不是整数的话要保留所有整数和小数，不要翻译任何公司或人名
然后一步一步获得最终答案
""",
        use_chat=False)
        #llm = self.langchainLib.get_llm(model = "llama3-groq-70b-8192-tool-use-preview")
        #llm = self.langchainLib.get_llm(model = "llama3-70b-8192")
        llm = self.langchainLib.get_llm(key = "LLM.TOGETHER")
        chain = prompt | llm
        response = chain.invoke({"input":content})
        return {"messages":[response],"ask_human":False}

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
        workflow.add_node("translate",self.translate_to_en)
        workflow.add_node("chatbot",self.chatbot)
        workflow.add_node("tools",ToolNode(tools=tools))
        workflow.add_node("human", self.human_node)
        
        workflow.add_conditional_edges("chatbot", self.select_next_node,
                                       {"human":"human","tools":"tools","__end__":"__end__"})
        
        workflow.add_edge(START, "translate")
        workflow.add_edge("translate","chatbot")
        workflow.add_edge("tools","chatbot")
        workflow.add_edge("human","chatbot")

        graph = workflow.compile(checkpointer=self.graphLib.memory,
                                 interrupt_before= ["human"])

        return graph    
