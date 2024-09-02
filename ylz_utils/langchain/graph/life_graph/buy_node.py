from  .state import Buys,State
from .node import Node

from langchain_core.messages import AIMessage
class BuyNode(Node):
    def buyNode(self,state:State):
        llm = self.get_llm()
        llm_with_output = llm.with_structured_output(Buys)
        message = state["messages"][-1]
        prompt = self.lifeGraph.graphLib.langchainLib.get_prompt()
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        res = (prompt | llm_with_output).invoke({"input":subTag.sub_text})
        if isinstance(res,Buys):
            self.create_record(res)    
            return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
        else:
            return {"messages":[AIMessage(content="抱歉,我无法解析饮食数据")],"life_tag":state["life_tag"]}  
    
    def create_record(self,buys:Buys):
        neo4jLib = self.lifeGraph.graphLib.langchainLib.neo4jLib
        # 处理时间问题
        for buy in buys.buys:
           buy.sdt ,buy.edt, buy.duration = self.parse_time(buy.sdt,buy.edt,buy.duration) 

        buys_list = buys.dict()["buys"]
        user_id = self.lifeGraph.user_id 
        script = """
            unwind $buys as buy
            match (n:Person{name:$user_id})
            call{            
                with buy with buy
                where buy.name is not null and buy.name <> ""
                merge (pd:Product{name:buy.name})
                return pd
            }
            call{
                with buy,pd
                with buy,pd
                where buy.brand is not null and buy.brand <> ""
                merge (b:Brand{name:buy.brand})
                merge (b)-[r:product]->(pd) 
            }
            call{
                with buy,n with buy,n
                where buy.place is not null and buy.place <> ""
                merge (p:Place{name:buy.place})
                create (p)<-[r:at{sdt:buy.sdt,edt:buy.edt}]-(n) 
            }
            call{
                with buy,n,pd
                create (n)-[r:buy{sdt:buy.sdt,edt:buy.edt,duration:buy.duration,place:buy.place,act:buy.act,name:buy.name,value:buy.value,unit:buy.unit,buy:buy.buy}]->(pd)
            }
        """
        neo4jLib.run(script,buys=buys_list,user_id=user_id)