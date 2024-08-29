from  .state import Diets,State
from .node import Node

from langchain_core.messages import AIMessage
class DietNode(Node):
    def dietNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Diets)
        message = state["messages"][-1]
        prompt = self.lifeGraph.graphLib.langchainLib.get_prompt()
        res = (prompt | llm_with_output).invoke({"input":message.content})
        if isinstance(res,Diets):
            self.create_record(res)
            return {"messages":[AIMessage(content=str(res))]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析饮食数据")]}  
    
    def create_record(self,diets:Diets):
        neo4jLib = self.lifeGraph.graphLib.langchainLib.neo4jLib
        diets_list = diets.dict()["foods"]
        user_id = self.lifeGraph.user_id 
        script = f"""
            unwind $diets as diet
            match (n:Person{{name:$user_id}})
            merge (m:Food{{name:diet.name}})
            create (n)-[r:diet{{sdt:diet.sdt,edt:diet.edt,duration:diet.duration,place:diet.place,act:diet.act,value:diet.value,unit:diet.unit}}]->(m)
        """
        neo4jLib.run(script,diets=diets_list,user_id=user_id)