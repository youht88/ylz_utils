from typing import List, Optional
from ylz_utils.file import FileLib
from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage,HumanMessage

from ylz_utils.langchain.graph.stock_graph import StockGraph
from .configurable import ConfigSchema
from .function import FunctionGraph
from ..public_graph.summary import SummaryGraph
from ..stock_graph.state import *
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
        #data = toolLib.get_company_info("ST易联众")
        #self.stockData = toolLib.get_hsmy_jddxt("瑞芯微")
    def load_data(self,func,csv_file_name,time_str,trans_to_datamodel:Optional[BaseModel]=None) ->pd.DataFrame|BaseModel:
        # self.data = self.load_data(self.toolLib.get_higg_jlr,"higg_jlr.csv","t",trans_to_datamodel=JLR)
        try:
            df = pd.read_csv(csv_file_name)
            given_date = datetime.strptime(df.loc[0][time_str], "%Y-%m-%d %H:%M:%S")
            # 获取当前日期
            now = datetime.now()
            # 创建当前日期的下午 4 点钟的 datetime 对象
            newdata_today = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if given_date.date() < now.date() and now > newdata_today:
                data = func()
                df = pd.DataFrame([item.model_dump() for item in data])
                df.to_csv(csv_file_name) 
            else:
                print(f"正在转换{csv_file_name}")
                data = [trans_to_datamodel(**row) for index, row in df.iterrows()]  
                print("转换完毕!")
        except Exception as e:
            print("error on retrieve data,now re retrieve",e)
            data = func()
            df = pd.DataFrame([item.model_dump() for item in data])
            df.to_csv(csv_file_name)
        print(f"{csv_file_name} counts=",len(data))
        return data
    
    def get_graph(self) -> CompiledStateGraph:
        self.hslt_list = self.load_data(self.toolLib.get_hslt_list,"hslt_list.csv","t",trans_to_datamodel=HSLT_LIST)
        self.jlr = self.load_data(self.toolLib.get_higg_jlr,"higg_jlr.csv","t",trans_to_datamodel=JLR)
        self.gnbk = self.load_data(self.toolLib.get_hibk_gnbk,"hibk_gnbk.csv","t",trans_to_datamodel=GNBK) 
        self.zjhhy = self.load_data(self.toolLib.get_hibk_zjhhy,"hibk_zjhhy.csv","t",trans_to_datamodel=ZJHHY) 
        workflow = StateGraph(State,ConfigSchema)
        #workflow.add_node("function",FunctionGraph(self.langchainLib).get_graph())
        #workflow.add_node("summary",SummaryGraph(self.langchainLib).get_graph())
        #workflow.add_edge(START,"function")
        #workflow.add_edge("function","summary")
        #workflow.add_edge("summary",END)
        workflow.add_node("agent",self.agent0)
        workflow.add_edge("agent",END)
        workflow.add_edge(START,"agent")
        # workflow.add_node("exec",self.python_exec_tool)
        # workflow.add_edge("agent","exec")
        # workflow.add_conditional_edges("exec",self.if_continue)
        graph = workflow.compile(self.memory)
        return graph
        #return SummaryGraph(self.langchainLib).get_graph()
    def human_action(self, graph, config=None, thread_id=None) -> bool:
        return super().human_action(graph, config, thread_id)
    def agent0(self,state,config:RunnableConfig):
        #modelData = {"hslt_list":self.hslt_list,"jlr":self.jlr,"gnbk":self.gnbk,"zjhhy":self.zjhhy}
        llm = self.get_llm(config=config)
        llm_bind_tool = llm.bind_tools(self.get_class_instance_tools(self.toolLib))
        res = llm_bind_tool.invoke(state["messages"])
        print(res)
        return {"error":""}
    def python_exec_tool(self,state,config:RunnableConfig):
        #modelData:dict[str,BaseModel|list[BaseModel]] = state.get("modelData",{})
        modelData = {"hslt_list":self.hslt_list,"jlr":self.jlr,"gnbk":self.gnbk,"zjhhy":self.zjhhy}
        llm = self.get_llm(config=config)
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
        pattern = re.compile(r".*```python(.*)```",re.DOTALL)
        match = re.match(pattern,res.content)
        if not match:
            print("NO RESULT",res.content)
            return {"error":""}
        script = match.groups(0)[0]
        try:
            result={}
            exec(script,dataFrames,result)
            print("script--->",script)
            print("result--->",result)
        except Exception as e:
            print("ERROR!!",e,script)
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
