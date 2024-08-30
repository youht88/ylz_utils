from  .state import Diets,State
from .node import Node
import time,datetime

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
        # 处理时间问题
        print("food===>",diets.foods)
        for diet in diets.foods:
            parse_sdt=None
            parse_edt=None
            try:
                parse_sdt = self.jionlp.parse_time(diet.sdt,time_base=time.time())
                print("sdt:",parse_sdt)
            except Exception as e:
                pass
            try:
                parse_edt = self.jionlp.parse_time(diet.edt,tiem_base=time.time())    
                print("edt:",parse_edt)
            except Exception as e:
                pass
            if not parse_sdt and not parse_edt:
                diet.sdt = datetime.datetime.now()
                diet.edt = datetime.datetime.now()
            elif parse_sdt and not parse_edt:
                diet.sdt = datetime.datetime.fromisoformat(parse_sdt['time'][0])
                diet.edt = datetime.datetime.fromisoformat(parse_sdt['time'][1])
            elif parse_edt and not parse_sdt:
                diet.sdt = datetime.datetime.fromisoformat(parse_edt['time'][0])
                diet.edt = datetime.datetime.fromisoformat(parse_edt['time'][1])
            elif parse_edt and parse_sdt:
                diet.sdt = datetime.datetime.fromisoformat(parse_sdt['time'][0])
                diet.edt = datetime.datetime.fromisoformat(parse_edt['time'][1])

        diets_list = diets.dict()["foods"]
        print("new food--->",diets_list)
        user_id = self.lifeGraph.user_id 
        script = """
            unwind $diets as diet
            match (n:Person{name:$user_id})
            
            with diet,n
            where diet.name is not null
            merge (m:Food{name:diet.name})
            
            with diet,n,m
            optional match(:Brand{name:diet.brand}) 
            where diet.brand is not null
            merge (b:Brand{name:diet.brand})

            with diet,n,m,b
            where b is not null
            merge (b)-[r:product]->(m) 
            
            with diet,n,m
            optional match(:Place{name:diet.place})
            where diet.place is not null
            merge (p:Place{name:diet.place})
            
            with diet,n,m,p
            where p is not null
            merge (p)<-[r:at{sdt:diet.sdt,edt:diet.edt}]-(n) 

            with diet,n,m
            create (n)-[r:diet{sdt:diet.sdt,edt:diet.edt,duration:diet.duration,place:diet.place,act:diet.act,value:diet.value,unit:diet.unit}]->(m)
        """
        neo4jLib.run(script,diets=diets_list,user_id=user_id)