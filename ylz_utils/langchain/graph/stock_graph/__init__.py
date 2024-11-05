import json
from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage,ToolMessage

from .configurable import ConfigSchema
from .state import *
from ylz_utils.stock import StockLib
from ylz_utils.stock.mairui.mairui_hibk import HIBK
from ..public_graph.summary import SummaryGraph

from rich import print
from datetime import datetime

class StockGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        #toolsLib = Tools(self)
        #self.tools = [toolsLib.pankou,toolsLib.quotec,toolsLib.income,
        #              toolsLib.holders,toolsLib.balance,toolsLib.top_holders]
        self.tools:list = self.get_class_instance_tools(HIBK)
        self.tools.append(self.python_repl_tool)
    def get_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(NewState,ConfigSchema)
        workflow.add_node("summary_convasation",self.summarize_conversation)
        workflow.add_node("agent",Agent(self))
        workflow.add_node("tools",ToolNode(tools=self.tools))
        workflow.add_edge(START,"summary_convasation")
        workflow.add_edge("summary_convasation","agent")
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
        #print(llm_bind_tools)
        summary = state.get("summary","")
        systemPrompt = ("你是个人信息助理。调用相应的股票相关函数查找和分析数据"
                  "- 不要编造任何信息，仅记录我提供给你的信息。"
                  "- 不要产生幻觉"
                  "- 当前日期:{today}"
                  "- 之前对话的总结为:{summary}")
        messages = [SystemMessage(systemPrompt.format(today=datetime.now(),summary=summary))] + state["messages"]
           
        res = llm_bind_tools.invoke(messages)
        res = self.graphLib.get_safe_response(res)
        if isinstance(state["messages"][-1],ToolMessage):
            try:
                content = state["messages"][-1].content
                print("[DEBUG0]",content)
                print("???",state["messages"][-1])
                content = json.loads(content)
                print("[DEBUG1]",content)
                #content = MMWP.model_validate(content)
                #print("[DEBUG2]",content)
                return {"mmwp":content,"messages":[res]}
            except Exception as e:
                print("[ERROR]",e)
        else:
            return {"mmwp":{},"messages":[res]}
        
