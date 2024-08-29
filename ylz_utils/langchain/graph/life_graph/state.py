from langchain_core.pydantic_v1 import BaseModel,Field
from typing import List,Optional,Literal

from langgraph.graph import MessagesState

class Tag(BaseModel):
    '''
    标记句子的意图及类型并为分类标记确信度
    举例:
    我吃了一个苹果 -> is_question=fale,action=record,type=diet
    我下午3点要开会 -> is_question=false,action=schedule,type=other
    我解析来一个礼拜要每天跑2公里 -> is_question=false,action=plan,type=sport
    体重70公斤，身高1.74 -> is_question=false,action=recored,type=sign
    怎样学好数学？ -> is_question=true,action=other,type=other
    练习打羽毛球并没有那么容易 -> is_question=false,action=other,type=sport
    '''
    is_question: bool = Field(description="是否为提问语句")
    action: Literal["record","plan","schedule","other"] = Field(description="语句意图是记录、计划还是日程安排")
    type: Literal["diet","sport","sign","other"] = Field(description="分析句子是关于饮食、运动、体征还是其他")
    score: int = Field(description="为分类判断的置信度从1-5打分",max=5,min=1)

class State(MessagesState):
    life_tag: Tag
    human:bool

class Diet(BaseModel):
    '''解析为饮食的相关数据'''
    sdt:Optional[str] = Field(description="吃食品的开始时间")
    edt:Optional[str] = Field(description="吃食品的结束时间")
    duration: Optional[str] = Field(description="吃食品的时长,如2个小时、3分钟等")
    act:Optional[str] = Field(description="吃食品的动作,例如:吃、喝、口服等")
    place:Optional[str] = Field(description="吃食品的地点")
    name:str = Field(description="食品的名称")
    value:float = Field(description="食品的数量")
    unit: str = Field(description="食品的数量")

class Diets(BaseModel):
    foods:List[Diet] = Field(description="一组食品")

class Sport(BaseModel):
    '''解析为运动的数据'''
    sdt:Optional[str] = Field(description="运动的开始时间")
    edt:Optional[str] = Field(description="运动的结束时间")
    duration: Optional[str] = Field(description="运动的时长,如2个小时、3分钟等")
    act:Optional[str] = Field(description="运动的动作,例如:打、跑、踢、跳等")
    place:Optional[str] = Field(description="运动的地点")
    name:str = Field(description="运动的名称")
    distance: Optional[float] = Field(description="运动的距离")

class Sports(BaseModel):
    sports:List[Sport] = Field(description="一组运动")

class Sign(BaseModel):
    '''解析为体征测量数据'''
    sdt:Optional[str] = Field(description="测量的开始时间")
    edt:Optional[str] = Field(description="测量的结束时间")
    duration: Optional[str] = Field(description="测量的时长,如2个小时、3分钟等")
    act:Optional[str] = Field(description="测量的动作,例如:测量等")
    place:Optional[str] = Field(description="测量的地点")
    name:str = Field(description="体征的名称")
    value: float = Field(description="体征结果的数值")
    unit: str = Field(description="体征结果数值的单位")

class Signs(BaseModel):
    signs:List[Sign] = Field(description="一组体征")
