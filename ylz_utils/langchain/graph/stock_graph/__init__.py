from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from .configurable import ConfigSchema
from .state import State
from .tools import Tools

from rich import print
from datetime import datetime

class StockGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        toolsLib = Tools(self)
        self.tools = [toolsLib.pankou,toolsLib.balance]
    def get_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(MessagesState,ConfigSchema)
        workflow.add_node("agent",Agent(self))
        workflow.add_node("tools",ToolNode(tools=self.tools))
        workflow.add_edge(START,"agent")
        workflow.add_conditional_edges("agent",tools_condition)
        workflow.add_edge("tools","agent")
        graph = workflow.compile(self.memory)
        return graph

    def human_action(self, graph, config=None, thread_id=None) -> bool:
        return super().human_action(graph, config, thread_id)

class Agent():
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib                
    def __call__(self,state,config:RunnableConfig):
        llm_key = config.get("configurable",{}).get("llm_key")
        llm_model = config.get("configurable",{}).get("llm_model")
        if llm_key or llm_model:
            llm = self.graphLib.langchainLib.get_llm(llm_key,llm_model)
        else:
            llm = self.graphLib.get_node_llm()        
        llm_bind_tools = llm.bind_tools(self.graphLib.tools)
        print(llm_bind_tools)
        systemPrompt = ("你是个人信息助理。调用相应的股票相关函数查找数据"
                  "1、不要编造任何信息，仅记录我提供给你的信息。"
                  "2、不要产生幻觉"
                  "3、当前日期:{today}")
        messages = [SystemMessage(systemPrompt.format(today=datetime.now()))] + state["messages"]
        res = llm_bind_tools.invoke(messages)
        res = self.graphLib.get_safe_response(res)
        return {"messages":[res]}