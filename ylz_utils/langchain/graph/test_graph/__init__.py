from typing import List
from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage,HumanMessage

from ylz_utils.langchain.graph.stock_graph import StockGraph
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools
from .configurable import ConfigSchema
from .function import FunctionGraph
from ..public_graph.summary import SummaryGraph
from ..stock_graph.state import JDDXT,JLR,ZLJLR,SHJLR
from pydantic import BaseModel,Field
from rich import print
import pandas as pd

class State(MessagesState):
    command:str

class Info(BaseModel):
    '''公司信息'''
    dm:str = Field(description="代码")
    mc:str = Field(description="简称")

class TestGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        self.info: List[Info] = [Info(dm='603893',mc='瑞芯微')]
        stockGraph = StockGraph(langchainLib)
        toolLib = MairuiTools(stockGraph)
        #data = toolLib.get_company_info("ST易联众")
        #self.stockData = toolLib.get_hsmy_jddxt("瑞芯微")
        self.jlr = toolLib.get_higg_jlr()
    def get_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(State,ConfigSchema)
        #workflow.add_node("function",FunctionGraph(self.langchainLib).get_graph())
        #workflow.add_node("summary",SummaryGraph(self.langchainLib).get_graph())
        #workflow.add_edge(START,"function")
        #workflow.add_edge("function","summary")
        #workflow.add_edge("summary",END)
        workflow.add_node("agent",self.agent)
        workflow.add_edge(START,"agent")
        graph = workflow.compile(self.memory)
        return graph
        #return SummaryGraph(self.langchainLib).get_graph()
    def human_action(self, graph, config=None, thread_id=None) -> bool:
        return super().human_action(graph, config, thread_id)
    
    def agent(self,state,config:RunnableConfig):
        llm_key = config.get('configurable',{}).get('llm_key')
        llm_model = config.get('configurable',{}).get('llm_model')
        llm = self.langchainLib.get_llm(llm_key,llm_model)
        prompt = ("根据一组数据结构的上下文dataframe_description信息，结合用户提示生成需要执行的python程序。\n"
                  "不要假设数据，数据类型为dataframe。\n"
                  "其中`公司信息`变量名为info,"
                  "`净流入`变量名为jlr,结果必须存储在RESULT变量中。\n"
                  "请仅返回程序代码,不要包括```python\n"
                  "dataframe_description:{context}"
        )
        lastmessage = state["messages"][-1]
        print("stock data length:",self.jlr[:10])
        #print("result:",eval(script))
        context = "\n".join([str(Info.model_json_schema()),str(JLR.model_json_schema())])
        res = llm.invoke([SystemMessage(prompt.format(context = context)),lastmessage])
        script = res.content
        print(script)
        result={}
        exec(script,{"info":pd.DataFrame([item.model_dump() for item in self.info]),
                    "jlr":pd.DataFrame([item.model_dump() for item in self.jlr])},result)
        print(result)
        return {"messages":[res]}