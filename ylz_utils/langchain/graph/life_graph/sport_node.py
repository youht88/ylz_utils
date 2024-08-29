from  .state import Sports,State
from .node import Node

from langchain_core.messages import AIMessage
class SportNode(Node):
    def sportNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Sports)
        message = state["messages"][-1]
        prompt = self.lifeGraph.graphLib.langchainLib.get_prompt()
        res = (prompt | llm_with_output).invoke({"input":message.content})
        if isinstance(res,Sports):
            self.create_record(res)
            return {"messages":[AIMessage(content=str(res))]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析运动数据")]}  
    
    def create_record(self,sports:Sports):
        neo4jLib = self.lifeGraph.graphLib.langchainLib.neo4jLib
        sports_list = sports.dict()["sports"]
        user_id = self.lifeGraph.user_id 
        script = f"""
            unwind $sports as sport
            match (n:Person{{name:$user_id}})
            merge (m:Sport{{name:sport.name}})
            create (n)-[r:sport{{sdt:sport.sdt,edt:sport.edt,duration:sport.duration,
            place:sport.place,act:sport.act,value:sport.value,unit:sport.unit,value:sport.value,unit:sport.unit}}]->(m)
        """
        neo4jLib.run(script,sports=sports_list,user_id=user_id)