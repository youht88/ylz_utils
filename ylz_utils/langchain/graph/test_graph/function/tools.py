from __future__ import annotations
from typing import TYPE_CHECKING, Annotated, Optional
if TYPE_CHECKING:
    from . import TestGraph
from rich import print   
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

class Address(BaseModel):
    home_address:str = Field(description="家庭住址")
    work_address:str = Field(description="工作地址")

class UserInfo(BaseModel):
    id:str = Field(description="用户ID")
    name:str = Field(description="姓名")
    birthday:Optional[str] = Field(description="出生日期(YYYY-MM-DD)",examples=["1975-03-05","2002-08-25"])
    phone:str = Field(description="电话号码,不允许为空串")
    unit:Optional[str] = Field(description="工作单位")
    address:Optional[Address] = Field(description="家庭及工作地址")
    couple:Optional[str] = Field(description="配偶个人ID")

usersInfo:dict[str,UserInfo] = {}

class Tools:
    def __init__(self,graphLib:TestGraph):
        self.graphLib = graphLib
        self.graphLib.tools = [self.get_user_info,self.set_user_info,self.del_user_info]
    
    def get_user_info(self,config:RunnableConfig) -> UserInfo:
        '''
        获得用户信息。当用户询问自己的信息的时候调用此函数
        '''
        user_id = config.get("configurable",{}).get("user_id")
        if user_id in usersInfo:
            return usersInfo.get(user_id)
        else:
            return f"没有找到{user_id}的信息"     

    def set_user_info(self,info:UserInfo,config:RunnableConfig) -> None:
        '''
        设置用户信息。当用户告诉我他的相关信息的时候调用此函数
        '''
        user_id = config.get("configurable",{}).get("user_id")
        usersInfo[user_id] = info  
 
    def del_user_info(self,config:RunnableConfig, state: Annotated[dict, InjectedState]) -> None:
        '''
        删除用户信息。当用户希望删除或忘记他的个人信息时调用此函数
        '''
        print("DEBUG STATE",len(state["messages"]),state["messages"][-1])
        if len(state["messages"])>10:
            raise ValueError("交互的内容太多啦！！！！")

        user_id = config.get("configurable",{}).get("user_id")
        if user_id in usersInfo:
            del usersInfo[user_id]  
