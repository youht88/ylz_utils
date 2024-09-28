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
            print(llm)
        else:
            llm = self.graphLib.get_node_llm()
        llm_bind_tools = llm.bind_tools(self.graphLib.tools)
        systemPrompt = ("你是个人信息助理。调用相应的函数查找或设置个人信息，"
                  "不要编造任何信息，仅记录我提供给你的信息。"
                  "不要产生幻觉"
                  "当前日期:{today}")
        messages = [SystemMessage(systemPrompt.format(today=datetime.now()))] + state["messages"]
        
        response  = llm_bind_tools.invoke(messages)
        response = self.graphLib.get_safe_response(response)
        return {"messages":[response]}