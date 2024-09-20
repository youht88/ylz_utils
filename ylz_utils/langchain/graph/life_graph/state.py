from pydantic import BaseModel,Field
from typing import List,Optional,Literal, Union

from langgraph.graph import MessagesState

class SubTag(BaseModel):
    """
    拆分句子后得到的类型标记。
    diet包括各种饮食，如牛奶、苹果、西红柿炒蛋
    sport包括各种运动，如打篮球、打乒乓、跑步、俯卧撑
    sign包括身体体征，如身高、体重、血压、心率
    buy包括各种购物行为
    """
    sub_text: str = Field(description="相关的子句,注意：需要理解原句子的意思，适当的时候对字句补充相关时间、地点、花费的信息")
    type: Literal["diet","sport","sign","buy","other"] = Field(description="分析句子是关于饮食、运动、体征、购物还是其他")
class Tag(BaseModel):
    '''
    标记句子的意图及类型,并为可以直接给出问题答案的程度打分。要求：
    1、不要遗漏日期、地点等关键条件
    2、如果没有指定日期，且根据上下文也无法判别日期或日期范围，则默认为当天
    举例:
    我吃了一个苹果 -> is_question=fale,action=record,type=diet
    我下午3点要开会 -> is_question=false,action=schedule,type=other
    我解析来一个礼拜要每天跑2公里 -> is_question=false,action=plan,type=sport
    体重70公斤，身高1.74 -> is_question=false,action=recored,type=sign
    怎样学好数学？ -> is_question=true,action=other,type=other
    练习打羽毛球并没有那么容易 -> is_question=false,action=other,type=sport
    下午3点打篮球打到6点半 -> is_question=false,action=record,type=sport
    昨天下午在大润发花了25块3毛六买了3个电风扇 -> is_question=false,action=record,type=buy
    我今天买了什么？ -> is_question=true,action=other,type=buy
    近三天买笔花了多少？ -> is_question=true,action=other,type=buy
    这个礼拜做了哪些运动？-> is_question=true,action=other,type=sport
    这个月平均体重是多少 -> is_question=true,action=other,type=sign 
    今天体重多少公斤？ -> is_question=true,action=other,type=sign
    上周3吃了什么？ -> is_question=true,action=other,type=diet
    除了火龙果，我还吃了什么？-> is_question=true,action=other,type=diet
    '''
    is_question: bool = Field(description="是否为提问语句")
    action: Literal["record","plan","schedule","other"] = Field(description="语句意图是记录(已经完成)、计划(准备完成的目标)还是日程安排(准备完成的安排),如果都不是则为其他")
    subTags: List[SubTag] = Field(description="将语句的原始意思进行拆分。注意：需要理解原句子的意思，适当的时候对字句补充相关时间、地点、花费的信息")
    score: int = Field(description="为可以直接给出问题的答案的可能性打分",min=1,max=5)

class State(MessagesState):
    life_tag: Tag
    human:bool

class Diet(BaseModel):
    '''解析为饮食的相关数据'''
    sdt:Optional[str] = Field(description="吃食品的开始时间,比如上周末、2个小时前、昨天下午2点等")
    edt:Optional[str] = Field(description="吃食品的结束时间,比如上周末、2个小时前、昨天下午2点等")
    duration: Optional[str] = Field(description="吃食品的时长,以分钟单位。如2个小时、3分钟等")
    act:Optional[str] = Field(description="吃食品的动作,例如:吃、喝、口服等")
    place:Optional[str] = Field(description="吃食品的地点")
    name:str = Field(description="食品的名称")
    value:float = Field(description="食品的数量")
    unit: str = Field(description="食品的数量的单位")
    brand: Optional[str] = Field(description="食物的品牌,比如:匹克薯片->brand:匹克,麦当劳汉堡->brand:麦当劳。如果没有指定则为空")
    buy: Optional[float]= Field(description="买食物花的钱，比如：3块5")
    cal: float = Field(description="估算这些数量食物的热量，单位是卡路里。例如:如果一个苹果95卡,那么3个苹果为285卡")
class Diets(BaseModel):
    foods:List[Diet] = Field(description="一组食品")

class Sport(BaseModel):
    '''解析为运动的数据'''
    sdt:Optional[str] = Field(description="运动的开始时间,比如上周末、2个小时前、昨天下午2点等")
    edt:Optional[str] = Field(description="运动的结束时间,比如上周末、2个小时前、昨天下午2点等")
    duration: Optional[str] = Field(description="运动的时间间隔,以分钟单位。比如2个小时、3分钟等")
    act:Optional[str] = Field(description="运动的动作,比如:打、跑、踢、跳等")
    place:Optional[str] = Field(description="运动的地点")
    name:str = Field(description="运动的名称")
    value: Optional[float] = Field(description="运动的数量,比如300次,150米等")
    unit: Optional[str] = Field(description="运动的数量的单位")
    buy: Optional[float]= Field(description="本次运动所花的钱,比如:3块5")
    cal: float = Field(description="估算这些数量或运动时长的运动消耗的热量，单位是卡路里")
class Sports(BaseModel):
    sports:List[Sport] = Field(description="一组运动")

class Sign(BaseModel):
    '''解析为体征测量数据，体征数据包括身高、体重、血压、心率、视力、胸围等'''
    sdt:Optional[str] = Field(description="测量的开始时间，比如上周末、2个小时前、昨天下午2点等")
    edt:Optional[str] = Field(description="测量的结束时间，比如上周末、2个小时前、昨天下午2点等")
    duration: Optional[str] = Field(description="测量的时长,如2个小时、3分钟等")
    act:Optional[str] = Field(description="测量的动作,例如:测量等")
    place:Optional[str] = Field(description="测量的地点")
    name:str = Field(description="体征的名称")
    value: Union[float,str] = Field(description="体征结果的数值，如:30、70.7;或者询问体征的内容，如:最大多少、平均数")
    unit: str = Field(description="体征结果数值的单位，如果没有指定则根据名称默认为常识单位")
    buy: Optional[float]= Field(description="本次测量所花的钱,比如:3块5")

class Signs(BaseModel):
    signs:List[Sign] = Field(description="一组体征")

class Buy(BaseModel):
    '''解析为购买商品的相关数据'''
    sdt:Optional[str] = Field(description="购物的开始时间,比如上周末、2个小时前、昨天下午2点等")
    edt:Optional[str] = Field(description="购物的结束时间,比如上周末、2个小时前、昨天下午2点等")
    duration: Optional[str] = Field(description="购物的时长,以分钟单位。如2个小时、3分钟等")
    act:Optional[str] = Field(description="购物的动作,例如:买、购买等")
    place:Optional[str] = Field(description="购物的地点")
    name:str = Field(description="物品的名称")
    value:float = Field(description="物品的数量")
    unit: str = Field(description="物品的数量的单位")
    brand: Optional[str] = Field(description="物品的品牌,比如:匹克薯片->brand:匹克,麦当劳汉堡->brand:麦当劳。如果没有指定则为空")
    buy: Optional[float]= Field(description="购物花的钱，比如：3块5")
class Buys(BaseModel):
    buys:List[Buy] = Field(description="一组购物")