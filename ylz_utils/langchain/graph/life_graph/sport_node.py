from  .state import Sports,State
from .node import Node

from langchain_core.messages import AIMessage
class SportNode(Node):
    def sportNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Sports)
        message = state["messages"][-1]
        prompt = self.lifeGraph.langchainLib.get_prompt()
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        res = (prompt | llm_with_output).invoke({"input":subTag.sub_text})
        if isinstance(res,Sports):
            self.create_record(res)    
            return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析运动数据")],"life_tag":state["life_tag"]}      

    def create_record(self,sports:Sports):
        # 处理时间问题
        for sport in sports.sports:
            sport.sdt ,sport.edt, sport.duration = self.parse_time(sport.sdt,sport.edt,sport.duration) 

        sports_list = sports.dict()["sports"]
        user_id = self.lifeGraph.user_id 
        script = """
            unwind $sports as sport
            match (n:Person{name:$user_id})
            call{            
                with sport with sport
                where sport.name is not null and sport.name <> ""
                merge (s:Sport{name:sport.name})
                return s
            }
            call{
                with sport,n with sport,n
                where sport.place is not null and sport.place <> ""
                merge (p:Place{name:sport.place})
                create (p)<-[r:at{sdt:sport.sdt,edt:sport.edt}]-(n) 
            }
            call{
                with sport,n,s
                create (n)-[r:sport{sdt:sport.sdt,edt:sport.edt,duration:sport.duration,place:sport.place,act:sport.act,name:sport.name,value:sport.value,unit:sport.unit,buy:sport.buy}]->(s)
            }
        """
        self.neo4jLib.run(script,sports=sports_list,user_id=user_id)