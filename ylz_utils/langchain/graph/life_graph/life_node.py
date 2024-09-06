from  .state import Diets,Sports,Signs,Buys,State
from .node import Node

from langchain_core.messages import AIMessage
class LifeNode(Node):
    def __init__(self,lifeGraph,msg=None):
        super().__init__(lifeGraph,msg)
        self.llm = self.graphLib.get_node_llm()
        self.llm_diet = self.llm.with_structured_output(Diets)
        self.llm_sport = self.llm.with_structured_output(Sports)
        self.llm_sign = self.llm.with_structured_output(Signs)
        self.llm_buy = self.llm.with_structured_output(Buys)
    def __call__(self,state:State):
        message = state["messages"][-1]
        prompt = self.graphLib.langchainLib.get_prompt()
        subTag = state["life_tag"].subTags[0]
        state["life_tag"].subTags = state["life_tag"].subTags[1:]
        tag_type = subTag.type
        match tag_type:
            case "diet":
                res = (prompt | self.llm_diet).invoke({"input":subTag.sub_text})
                if isinstance(res,Diets):
                    self.create_diet_record(res)    
                    return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我无法解析饮食数据")],"life_tag":state["life_tag"]}  
            case "sport":
                res = (prompt | self.llm_sport).invoke({"input":subTag.sub_text})
                if isinstance(res,Sports):
                    self.create_sport_record(res)    
                    return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我无法解析运动数据")],"life_tag":state["life_tag"]}  
            case "sign":
                res = (prompt | self.llm_sign).invoke({"input":subTag.sub_text})
                if isinstance(res,Signs):
                    self.create_sign_record(res)    
                    return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我无法解析体征测量的数据")],"life_tag":state["life_tag"]}  
            case "buy":
                res = (prompt | self.llm_buy).invoke({"input":subTag.sub_text})
                if isinstance(res,Buys):
                    self.create_buy_record(res)    
                    return {"messages":[AIMessage(content=str(res))],"life_tag":state["life_tag"]}
                else:
                    return {"messages":[AIMessage(content="抱歉,我无法解析购物数据")],"life_tag":state["life_tag"]}  
            case _:
                raise Exception("不可能到达这个节点")
        
    def create_diet_record(self,diets:Diets):
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
                create (n)-[r:diet{sdt:diet.sdt,edt:diet.edt,duration:diet.duration,
                    place:diet.place,act:diet.act,name:diet.name,value:diet.value,unit:diet.unit,buy:diet.buy,cal:diet.cal}]->(f)
            }
        """
        self.graphLib.neo4jLib.run(script,diets=diets_list,user_id=user_id)
    
    def create_sport_record(self,sports:Sports):
        # 处理时间问题
        for sport in sports.sports:
            sport.sdt ,sport.edt, sport.duration = self.parse_time(sport.sdt,sport.edt,sport.duration) 

        sports_list = sports.dict()["sports"]
        user_id = self.graphLib.user_id 
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
                create (n)-[r:sport{sdt:sport.sdt,edt:sport.edt,duration:sport.duration,
                    place:sport.place,act:sport.act,name:sport.name,value:sport.value,unit:sport.unit,buy:sport.buy,cal:sport.cal}]->(s)
            }
        """
        self.graphLib.neo4jLib.run(script,sports=sports_list,user_id=user_id)
    
    def create_sign_record(self,signs:Signs):
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
    
    def create_buy_record(self,buys:Buys):
        # 处理时间问题
        for buy in buys.buys:
           buy.sdt ,buy.edt, buy.duration = self.parse_time(buy.sdt,buy.edt,buy.duration) 

        buys_list = buys.dict()["buys"]
        user_id = self.graphLib.user_id 
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
        self.graphLib.neo4jLib.run(script,buys=buys_list,user_id=user_id)