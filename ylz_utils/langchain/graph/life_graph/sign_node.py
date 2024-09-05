from  .state import Signs,State
from .node import Node

from langchain_core.messages import AIMessage
class SignNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_with_output = self.llm.with_structured_output(Signs)
    def __call__(self,state:State):
        message = state["messages"][-1]
        prompt_template = "解析语句，不要出现幻觉！如果是疑问句,value为疑问的文本。如:体重是多少 -> value:value，最大身高是多少？-> value:max(value)"
        prompt = self.graphLib.langchainLib.get_prompt(prompt_template)
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        res = (prompt | self.llm_with_output).invoke({"input":subTag.sub_text})
        if isinstance(res,Signs):
            if state["life_tag"].is_question:
                res = self.query(res)
            else:
                self.create_record(res)    
            return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析体征数据")],"life_tag":state["life_tag"]}  
    
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