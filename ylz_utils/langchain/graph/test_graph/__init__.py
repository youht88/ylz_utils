from __future__ import annotations
from typing import TYPE_CHECKING,Literal,TypedDict,Annotated
if TYPE_CHECKING:
    from ylz_utils.langchain.graph import GraphLib

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage

from langchain_core.pydantic_v1 import BaseModel,Field

class State(MessagesState):
    ask_human: bool

class Output(BaseModel):
    originText:str = Field(description="原始文本")
    targetText:str = Field(description="翻译后的文本")
    len_target:int = Field(description="翻译后的文本长度")

class TestGraph():
    llm_key = None
    llm_model= None
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib
    def get_graph(self,llm_key,llm_model):
        self.llm_key = llm_key
        self.llm_model = llm_model
        workflow = StateGraph(State)
        workflow.add_node("robot",self.robot)
        workflow.add_edge(START,"robot")
        workflow.add_edge("robot",END)
        graph = workflow.compile(self.graphLib.memory)
        return graph        
    def robot(self,state:State):
        #outputParser = self.graphLib.langchainLib.get_outputParser(Output)
        #prompt = self.graphLib.langchainLib.get_prompt("把以下文本翻译成中文",outputParser = outputParser)
        prompt = self.graphLib.langchainLib.get_prompt("把以下文本翻译成中文")      
        llm = self.graphLib.langchainLib.get_llm(self.llm_key,self.llm_model)
        chain =  prompt | llm.with_structured_output(Output)
        message = state["messages"][-1]         
        translate_dict = chain.invoke({"input":message.content})
        print(translate_dict)
        
        return {"messages":[AIMessage(content = str(translate_dict.targetText))],"ask_human":False}
    