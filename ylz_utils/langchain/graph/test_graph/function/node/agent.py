from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .. import TestGraph
from rich import print  
from langchain_core.messages import SystemMessage 
from langchain_core.runnables import RunnableConfig
from datetime import datetime
class Agent:
    def __init__(self,graphLib:TestGraph):
        self.graphLib = graphLib
    def __call__(self,state,config:RunnableConfig):
        llm_key = config.get("configurable",{}).get("llm_key")
        llm_model = config.get("configurable",{}).get("llm_model")
        if llm_key or llm_model:
            llm = self.graphLib.langchainLib.get_llm(llm_key,llm_model)
        else:
            llm = self.graphLib.get_node_llm()
        print("debug tools:",self.graphLib.tools[-1])
        test_tools = self.graphLib.tools[-1]
        llm_bind_tools = llm.bind_tools(self.graphLib.tools)
        systemPrompt = ("你是个人信息助理。调用相应的函数查找或设置个人信息，"
                  "1、涉及金融相关的问题，请使用python repl的yfinance获取。执行脚本的最后一条语句务必使用print(...)返回结果，从而避免空数据"
                  "2、当使用yfinanace时要注意period参数只能是其中之一:['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']"
                  " 例如: 获取近10天的交易数据,stock.history(period='10d')是错误的，应该改为stock.history(period='1mo')后再截取;\n"
                  "      获取最近3个交易日的交易数据,stock.history(period='3d')是错误的，应该改为stock.history(period='5d')后再截取;\n"
                  "3、不要编造任何信息，仅记录我提供给你的信息。"
                  "4、不要产生幻觉"
                  "5、当前日期:{today}")
        messages = [SystemMessage(systemPrompt.format(today=datetime.now()))] + state["messages"]
        
        response  = llm_bind_tools.invoke(messages)
        response = self.graphLib.get_safe_response(response)
        return {"messages":[response]}