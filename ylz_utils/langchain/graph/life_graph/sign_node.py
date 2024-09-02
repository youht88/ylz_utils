from  .state import Signs,State
from .node import Node

from langchain_core.messages import AIMessage
class SignNode(Node):
    def signNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Signs)
        message = state["messages"][-1]
        prompt = self.lifeGraph.graphLib.langchainLib.get_prompt()
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        res = (prompt | llm_with_output).invoke({"input":subTag.sub_text})
        if isinstance(res,Signs):
            self.create_record(res)    
            return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析体征数据")],"life_tag":state["life_tag"]}  
    
    def create_record(self,signs:Signs):
        neo4jLib = self.lifeGraph.graphLib.langchainLib.neo4jLib
        # 处理时间问题
        for sign in signs.signs:
           sign.sdt ,sign.edt, sign.duration = self.parse_time(sign.sdt,sign.edt,sign.duration) 

        signs_list = signs.dict()["signs"]
        user_id = self.lifeGraph.user_id 
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
                create (n)-[r:sign{sdt:sign.sdt,edt:sign.edt,duration:sign.duration,place:sign.place,act:sign.act,value:sign.value,unit:sign.unit,buy:sign.buy}]->(s)
            }
        """
        neo4jLib.run(script,signs=signs_list,user_id=user_id)