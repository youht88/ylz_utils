from  .state import Diets,State
from .node import Node

from langchain_core.messages import AIMessage
class DietNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_with_output = self.llm.with_structured_output(Diets)
    def __call__(self,state:State):
        message = state["messages"][-1]
        prompt = self.graphLib.langchainLib.get_prompt()
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        res = (prompt | self.llm_with_output).invoke({"input":subTag.sub_text})
        if isinstance(res,Diets):
            self.create_record(res)    
            return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析饮食数据")],"life_tag":state["life_tag"]}  
    
    def create_record(self,diets:Diets):
        # 处理时间问题
        for diet in diets.foods:
           diet.sdt ,diet.edt, diet.duration = self.parse_time(diet.sdt,diet.edt,diet.duration) 

        diets_list = diets.dict()["foods"]
        user_id = self.graphLib.user_id 
        script = """
            unwind $diets as diet
            match (n:Person{name:$user_id})
            call{            
                with diet with diet
                where diet.name is not null and diet.name <> ""
                merge (f:Food{name:diet.name})
                return f
            }
            call{
                with diet,f
                with diet,f
                where diet.brand is not null and diet.brand <> ""
                merge (b:Brand{name:diet.brand})
                merge (b)-[r:product]->(f) 
            }
            call{
                with diet,n with diet,n
                where diet.place is not null and diet.place <> ""
                merge (p:Place{name:diet.place})
                create (p)<-[r:at{sdt:diet.sdt,edt:diet.edt}]-(n) 
            }
            call{
                with diet,n,f
                create (n)-[r:diet{sdt:diet.sdt,edt:diet.edt,duration:diet.duration,place:diet.place,act:diet.act,name:diet.name,value:diet.value,unit:diet.unit,buy:diet.buy}]->(f)
            }
        """
        self.graphLib.neo4jLib.run(script,diets=diets_list,user_id=user_id)