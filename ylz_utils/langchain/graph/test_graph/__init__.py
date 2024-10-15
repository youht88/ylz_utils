from typing import List
from ylz_utils.file import FileLib
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
from datetime import datetime
import re

class State(MessagesState):
    error:str
    modelData:dict[str,BaseModel|list[BaseModel]]

class TestGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        stockGraph = StockGraph(langchainLib)
        toolLib = MairuiTools(stockGraph)
        #data = toolLib.get_company_info("ST易联众")
        #self.stockData = toolLib.get_hsmy_jddxt("瑞芯微")
        try:
            jlr_df = pd.read_csv("higg_jlr.csv")
            given_date = datetime.strptime(self.jlr.loc[0]['t'], "%Y-%m-%d %H:%M:%S")
            # 获取当前日期
            now = datetime.now()
            # 创建当前日期的下午 4 点钟的 datetime 对象
            newdata_today = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if given_date.date() < now.date() and now > newdata_today:
                higg_jlr = toolLib.get_higg_jlr()
                self.jlr = higg_jlr
                jlr_df = pd.DataFrame([item.model_dump() for item in higg_jlr])
                jlr_df.to_csv("higg_jlr.csv") 
            else:
                print("888"*20)
                self.jlr = [JLR(**row) for index, row in jlr_df.iterrows()]  
        except Exception as e:
            higg_jlr = toolLib.get_higg_jlr()
            self.jlr = higg_jlr
            jlr_df = pd.DataFrame([item.model_dump() for item in higg_jlr])
            jlr_df.to_csv("higg_jlr.csv")
        print("higg jlr counts=",len(self.jlr))
    def get_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(State,ConfigSchema)
        #workflow.add_node("function",FunctionGraph(self.langchainLib).get_graph())
        #workflow.add_node("summary",SummaryGraph(self.langchainLib).get_graph())
        #workflow.add_edge(START,"function")
        #workflow.add_edge("function","summary")
        #workflow.add_edge("summary",END)
        workflow.add_node("agent",self.agent0)
        workflow.add_node("exec",self.python_exec_tool)
        workflow.add_edge(START,"agent")
        workflow.add_edge("agent","exec")
        workflow.add_conditional_edges("exec",self.if_continue)
        graph = workflow.compile(self.memory)
        return graph
        #return SummaryGraph(self.langchainLib).get_graph()
    def human_action(self, graph, config=None, thread_id=None) -> bool:
        return super().human_action(graph, config, thread_id)
    def agent0(self,state,config:RunnableConfig):
        modelData = {"jlr":self.jlr}
        return {"modelData":modelData}
    def python_exec_tool(self,state,config:RunnableConfig):
        modelData:dict[str,BaseModel|list[BaseModel]] = state.get("modelData",{})
        llm_key = config.get('configurable',{}).get('llm_key')
        llm_model = config.get('configurable',{}).get('llm_model')
        llm = self.langchainLib.get_llm(llm_key,llm_model)
        context1: list[str] = []
        context2: list[str] = []
        dataFrames ={}
        for key in modelData:
            if modelData[key]:
                modelData_key = modelData[key]
                if isinstance(modelData[key],list):
                    modelData_key=modelData[key][0]
                    dataFrames[key] = pd.DataFrame([item.model_dump() for item in modelData[key]])
                else:
                    dataFrames[key] = pd.DataFrame([modelData[key].model_dump()])
                context1.append(f"{key}")
                context2.append(f"{key}-->{modelData_key.model_json_schema()}")
                
        prompt0=(f"已知一组变量:{','.join(context1)}。每个变量都是pandas的dataframe对象\n"
                  f"各变量的结构定义如下:\n{'\n'.join(context2)}\n"
        )
        prompt = (f"{prompt0}"
                  "结合这些变量,根据用户提示生成需要执行的python程序。\n"
                  "不要假设数据，而是使用这些变量作为数据源\n"
                  "执行的最终结果必须赋值给`RESULT`变量\n"
                  "仅输出python代码,代码包含在```python```代码块中"
                  )
        print("[prompt]",prompt)
        lastmessage = state["messages"][-1]
        res = llm.invoke([SystemMessage(prompt)]+[lastmessage])
        match = re.match(r"```python(.*)```",res.content)
        print("????",match)
        if not match:
            print("NO RESULT",res.content)
            return {"error":""}
        script = match.groups(0)[0]
        print("[script]",script)
        try:
            result={}
            exec(script,dataFrames,result)
            print("script--->",script)
            print("result--->",result)
            res1 = llm.invoke([SystemMessage(f"{prompt0}")]+state["messages"]+[HumanMessage(str(result))])
            return {"error":"","messages":[res1]}
        except Exception as e:
            print("ERROR!!")
            return {"error":str(e)}
        
    def agent(self,state,config:RunnableConfig):
        llm_key = config.get('configurable',{}).get('llm_key')
        llm_model = config.get('configurable',{}).get('llm_model')
        llm = self.langchainLib.get_llm(llm_key,llm_model)
        context = "\n".join([f"JLR--->{JLR.model_json_schema()}\nJLR samples--->{self.jlr.sample(3)}"])
        prompt = ("根据一组数据结构的上下文dataframe_description信息，结合用户提示生成需要执行的python程序。\n"
                  "不要假设数据，数据类型为pandas dataframe。\n"
                  "数据变量采用以下指定的变量名，净流入数据变量名为jlr\n"
                  "结果必须存储在`RESULT`变量中。\n"
                  "请仅仅输出程序代码,不要包含```python\n"
                  "dataframe_description:{context}"
        ).format(context=context)
        print(f"|{prompt}|")
        lastmessage = state["messages"][-1]
        res = llm.invoke([SystemMessage(prompt)]+[lastmessage])
        print("!!!!",res)
        script = res.content
        try:
            result={}
            exec(script,{"jlr":self.jlr},result)
            print("script--->",script)
            print("result--->",result)
            res1 = llm.invoke([SystemMessage(f"数据结构说明:{context}")]+state["messages"]+[HumanMessage(str(result))])
            return {"error":"","messages":[res1]}
        except Exception as e:
            print("ERROR!!")
            return {"error":str(e)}
    def if_continue(self,state,config:RunnableConfig):
        if state["error"]:
            return "exec"
        else:
            return END
