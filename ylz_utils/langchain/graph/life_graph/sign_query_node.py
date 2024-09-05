import datetime
import re
from  .state import Signs,State
from .node import Node

from langchain_core.messages import AIMessage
class SignQueryNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_with_output = self.llm.with_structured_output(Signs)
    def __call__(self,state:State):
        message = state["messages"][-1]
        prompt_template = \
        """
        现在时间是:{now}
        根据neo4j的schema一步一步生成查询句子，确保生成neo4j查询语句script与查询的目标相关，且可以正确执行。
        要求1、仅输出script，不要做任何解释
           2、仅包含相关的节点和关系
           3、不要使用neo4j不存在的时间函数,使用date(sdt),date(edt)返回日期
        schema:{schema}
        """
        prompt = self.graphLib.langchainLib.get_prompt(prompt_template)
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        schema_description = """
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
        res = (prompt | self.llm).invoke({"now":datetime.datetime.now(),"schema":schema,"input":subTag.sub_text})
        print("query script1---->",res)
        res = re.findall("```cypher(.*)```",res.content.replace("\n"," "))
        if res:
            res = res[0]
            print("query script2---->",res)
            res = self.graphLib.neo4jLib.query(res)
            res = self.graphLib.neo4jLib.get_data(res)
        print("query script3---->",res)
        if res:
            return {"messages":[AIMessage(content=str(res))]}
        else:
            return {"messages":[AIMessage(content="抱歉,我没有统计出体征数据")]}  
    
    def query(self,signs:Signs):
        print("query?????",signs)
        # 处理时间问题
        for sign in signs.signs:
           sign.sdt ,sign.edt, sign.duration = self.parse_time(sign.sdt,sign.edt,sign.duration) 

        signs_list = signs.dict()["signs"]
        user_id = self.graphLib.user_id 
        script = """
            unwind $signs as sign
            match (n:Person{name:$user_id})
            match (n)-[r]-(s:Sign) where s.name = sign.name and datetime(sign.sdt) >= datetime(s.sdt) and datetime(sign.edt) <= datetime(s.edt) 
            return r.value
        """
        res = self.graphLib.neo4jLib.query(script,signs=signs_list[0],user_id=user_id)
        print("RESULT:",self.graphLib.neo4jLib.get_data(res))
    def create_record(self,signs:Signs):
        # 处理时间问题
        for sign in signs.signs:
           sign.sdt ,sign.edt, sign.duration = self.parse_time(sign.sdt,sign.edt,sign.duration) 

        signs_list = signs.dict()["signs"]
        user_id = self.graphLib.user_id 
        script = """
            unwind $signs as sign
            match (n:Person{name:$user_id})
            call{            
                with sign with sign
                where sign.name is not null and sign.name <> ""
                merge (s:Sign{name:sign.name})
                return s
            }
            call{
                with sign,n with sign,n
                where sign.place is not null and sign.place <> ""
                merge (p:Place{name:sign.place})
                create (p)<-[r:at{sdt:sign.sdt,edt:sign.edt}]-(n) 
            }
            call{
                with sign,n,s
                create (n)-[r:sign{sdt:sign.sdt,edt:sign.edt,duration:sign.duration,place:sign.place,act:sign.act,name:sign.name,value:sign.value,unit:sign.unit,buy:sign.buy}]->(s)
            }
        """
        self.graphLib.neo4jLib.run(script,signs=signs_list,user_id=user_id)