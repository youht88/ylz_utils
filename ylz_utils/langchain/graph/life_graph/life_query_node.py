import datetime
import re
from  .state import State,SubTag
from .node import Node

from langchain_core.messages import AIMessage
class LifeQueryNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
    def __call__(self,state:State):
        message = state["messages"][-1]
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        tag_type = subTag.type
        match tag_type:
            case 'diet':
                res = self.query_diet(subTag)
                if res:
                    return {"messages":[AIMessage(content=str(res))]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我没有统计出饮食数据")]}  
            case 'sport':
                res = self.query_sport(subTag)
                if res:
                    return {"messages":[AIMessage(content=str(res))]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我没有统计出运动数据")]}  
            case 'sign':
                res = self.query_sign(subTag)
                if res:
                    return {"messages":[AIMessage(content=str(res))]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我没有统计出体征测量的数据")]}  
            case 'buy':
                res = self.query_buy(subTag)
                if res:
                    return {"messages":[AIMessage(content=str(res))]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我没有统计出购物数据")]}  
            case _:
                raise Exception("不应该执行到这里") 
    def query_diet(self,subTag:SubTag):
        prompt_template = \
"""
现在时间是:{now}
我的名字是:{user_id}
根据neo4j的schema一步一步生成查询句子，确保生成neo4j查询语句script与查询的目标相关，且可以正确执行。
要求1、仅输出script，不要做任何解释
    2、仅包含相关的节点和关系
    3、不要使用neo4j不存在的时间函数,使用date(sdt),date(edt)返回日期
schema:{schema}
"""
        prompt = self.graphLib.langchainLib.get_prompt(prompt_template)
        schema_description = \
"""
(:Person).properties: <<name:姓名[String]>>
(:Food).properties: <<name:食物的名称[String]>>
(:Place).properties: <<name:地点名称[String]>>
[s:diet].properties: <<sdt:起始时间,edt:截止时间,place:吃东西的地点[String],cal:食物摄入的热量,value:食物的数量[Double],unit:食物数量的单位[String],buy:吃食物所花费的金额[Double],duration:吃食物所花费的时间[String]>>
[at].properties:<<sdt:起始时间,edt:截止时间>>
(:Person)-[:diet{{sdt,edt,place,value,unit,buy,duration,cal}}]->(:Food) 某人吃喝某些食物
(:Person)-[:at]->(:Place) 某人在某地
"""
        rel_schema = filter(lambda x:x.get("relType") in [":`diet`",":`at`"],self.graphLib.neo4jLib.get_node_schema())
        node_schema = filter(lambda x:x.get("nodeType") in [":`Person`",":`Food`",":`Place`"],self.graphLib.neo4jLib.get_node_schema())
        schema = f"schema description:\n{schema_description}\nrelationship schema:\n{rel_schema}\nnode schema:\n{node_schema}\n"
        res = (prompt | self.llm).invoke({"user_id":self.graphLib.user_id,"now":datetime.datetime.now(),"schema":schema,"input":subTag.sub_text})
        print("query script1---->",res)
        res = re.findall("```cypher(.*)```",res.content.replace("\n"," "))
        if res:
            res = res[0]
            print("query script2---->",res)
            res = self.graphLib.neo4jLib.query(res)
            res = self.graphLib.neo4jLib.get_data(res)
        print("query script3---->",res)
        return res
    def query_sport(self,subTag:SubTag):
        prompt_template = \
"""
现在时间是:{now}
我的名字是:{user_id}
根据neo4j的schema一步一步生成查询句子，确保生成neo4j查询语句script与查询的目标相关，且可以正确执行。
要求1、仅输出script，不要做任何解释
    2、仅包含相关的节点和关系
    3、不要使用neo4j不存在的时间函数,使用date(sdt),date(edt)返回日期
schema:{schema}
"""
        prompt = self.graphLib.langchainLib.get_prompt(prompt_template)
        schema_description = \
"""
(:Person).properties: <<name:姓名[String]>>
(:Sport).properties: <<name:运动的名称[String]>>
(:Place).properties: <<name:地点名称[String]>>
[s:diet].properties: <<sdt:起始时间,edt:截止时间,place:运动的地点[String],cal:运动消耗的热量[Double],value:某项运动的数量[Double],unit:某项运动数量的单位[String],buy:运动所花费的金额[Double],duration:运动所花费的时间[String]>>
[at].properties:<<sdt:起始时间,edt:截止时间>>
(:Person)-[:sport{{sdt,edt,place,value,unit,buy,duration,cal}}]->(:Sport) 某人完成某项运动
(:Person)-[:at]->(:Place) 某人在某地
"""
        rel_schema = filter(lambda x:x.get("relType") in [":`sport`",":`at`"],self.graphLib.neo4jLib.get_node_schema())
        node_schema = filter(lambda x:x.get("nodeType") in [":`Person`",":`Sport`",":`Place`"],self.graphLib.neo4jLib.get_node_schema())
        schema = f"schema description:\n{schema_description}\nrelationship schema:\n{rel_schema}\nnode schema:\n{node_schema}\n"
        res = (prompt | self.llm).invoke({"user_id":self.graphLib.user_id,"now":datetime.datetime.now(),"schema":schema,"input":subTag.sub_text})
        print("query script1---->",res)
        res = re.findall("```cypher(.*)```",res.content.replace("\n"," "))
        if res:
            res = res[0]
            print("query script2---->",res)
            res = self.graphLib.neo4jLib.query(res)
            res = self.graphLib.neo4jLib.get_data(res)
        print("query script3---->",res)
        return res
    def query_sign(self,subTag:SubTag):
        prompt_template = \
"""
现在时间是:{now}
我的名字是:{user_id}
根据neo4j的schema一步一步生成查询句子，确保生成neo4j查询语句script与查询的目标相关，且可以正确执行。
要求1、仅输出script，不要做任何解释
    2、仅包含相关的节点和关系
    3、不要使用neo4j不存在的时间函数,使用date(sdt),date(edt)返回日期
schema:{schema}
"""
        prompt = self.graphLib.langchainLib.get_prompt(prompt_template)
        schema_description = \
"""
(:Person).properties: <<name:姓名[String]>>
(:Sign).properties: <<name:体征的名称[String]>>
(:Place).properties: <<name:地点名称[String]>>
[s:sign].properties: <<sdt:起始时间,edt:截止时间,place:测量地点[String],value:测量的数据[Double],unit:测量的数据的单位[String],buy:测量所花费的金额[Double],duration:测量所花费的时间[String]>>
[at].properties:<<sdt:起始时间,edt:截止时间>>
(:Person)-[:sign{{sdt,edt,place,value,unit,buy,duration}}]->(:Sign) 某人测量某项身体指标
(:Person)-[:at]->(:Place) 某人在某地
"""
        rel_schema = filter(lambda x:x.get("relType") in [":`sign`",":`at`"],self.graphLib.neo4jLib.get_node_schema())
        node_schema = filter(lambda x:x.get("nodeType") in [":`Person`",":`Sign`",":`Place`"],self.graphLib.neo4jLib.get_node_schema())
        schema = f"schema description:\n{schema_description}\nrelationship schema:\n{rel_schema}\nnode schema:\n{node_schema}\n"
        res = (prompt | self.llm).invoke({"user_id":self.graphLib.user_id,"now":datetime.datetime.now(),"schema":schema,"input":subTag.sub_text})
        print("query script1---->",res)
        res = re.findall("```cypher(.*)```",res.content.replace("\n"," "))
        if res:
            res = res[0]
            print("query script2---->",res)
            res = self.graphLib.neo4jLib.query(res)
            res = self.graphLib.neo4jLib.get_data(res)
        print("query script3---->",res)
        return res

    def query_buy(self,subTag:SubTag):
        prompt_template = \
"""
现在时间是:{now}
我的名字是:{user_id}
根据neo4j的schema一步一步生成查询句子，确保生成neo4j查询语句script与查询的目标相关，且可以正确执行。
要求1、仅输出script，不要做任何解释
    2、仅包含相关的节点和关系
    3、不要使用neo4j不存在的时间函数,使用date(sdt),date(edt)返回日期
schema:{schema}
"""
        prompt = self.graphLib.langchainLib.get_prompt(human_prompt=prompt_template)
        schema_description = \
"""
(:Person).properties: <<name:姓名[String]>>
(:Product).properties: <<name:物品的名称[String]>>
(:Place).properties: <<name:地点名称[String]>>
[s:buy].properties: <<sdt:起始时间,edt:截止时间,place:买东西的地点[String],value:买的东西的数量[Double],unit:买东西的数量的单位[String],buy:买东西所花费的金额[Double],duration:买东西所花费的时间[String]>>
[at].properties:<<sdt:起始时间,edt:截止时间>>
(:Person)-[:diet{{sdt,edt,place,value,unit,buy,duration,cal}}]->(:Product) 某人购买某些物品
(:Person)-[:at]->(:Place) 某人在某地
"""
        rel_schema = filter(lambda x:x.get("relType") in [":`buy`",":`at`"],self.graphLib.neo4jLib.get_node_schema())
        node_schema = filter(lambda x:x.get("nodeType") in [":`Person`",":`Product`",":`Place`"],self.graphLib.neo4jLib.get_node_schema())
        schema = f"schema description:\n{schema_description}\nrelationship schema:\n{rel_schema}\nnode schema:\n{node_schema}\n"
        res = (prompt | self.llm).invoke({"user_id":self.graphLib.user_id,"now":datetime.datetime.now(),"schema":schema,"input":subTag.sub_text})
        res = re.findall("```cypher(.*)```",res.content.replace("\n"," "))
        if res:
            res = res[0]
            print("query script2---->",res)
            res = self.graphLib.neo4jLib.query(res)
            res = self.graphLib.neo4jLib.get_data(res)
        print("query script3---->",res)
        return res
