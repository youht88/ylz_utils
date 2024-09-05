
from typing import Literal
from ylz_utils.langchain.graph import GraphLib

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage


from .state import State, Tag
from .tag_node import TagNode
from .diet_node import DietNode
from .sport_node import SportNode
from .sign_node import SignNode
from .sign_query_node import SignQueryNode
from .buy_node import BuyNode
from .buy_query_node import BuyQueryNode
from .agent_node import AgentNode

class LifeGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        self.neo4jLib = self.get_neo4jLib()

    def get_neo4jLib(self):
        neo4jLib =  self.langchainLib.neo4jLib
        if not neo4jLib:
            raise Exception("请先调用langchainLib.init_neo4j(neo4j)")
        self.neo4jLib = neo4jLib
        return neo4jLib
    
    def human_action(self, graph, thread_id):
        return super().human_action(graph, thread_id)
    
    def router(self,state:State)->Literal["diet","sport","sign","buy","sign_query","buy_query","agent","__end__"]:
        tag:Tag = state["life_tag"]
        print("tag=======>",tag)
        if tag.is_question:
            for item in tag.subTags:
                match item.type:
                    case "diet":
                        return "agent"
                    case "sport":
                        return "agent"
                    case "sign":
                        return "sign_query"
                    case "buy":
                        return "buy_query"
                    case _:
                        return "agent"
            return "agent"
        match tag.action:
            case "record":
                for item in tag.subTags:
                    match item.type:
                        case "diet":
                            return "diet"
                        case "sport":
                            return "sport"
                        case "sign":
                            return "sign"
                        case "buy":
                            return "buy"
                        case _:
                            return "agent"
                return "agent"
            case _:
                return "agent"
    
    def get_graph(self):
        print("---> create graph:lifeGraph")
        workflow = StateGraph(State)
        
        workflow.add_node("tag",TagNode(self,"add node:tagNode"))
        workflow.add_node("diet",DietNode(self,"add node:dietNode"))
        workflow.add_node("sport",SportNode(self,"add node:sportNode"))
        workflow.add_node("sign",SignNode(self,"add node:signNode"))
        workflow.add_node("sign_query",SignQueryNode(self,"add node:signQueryNode"))
        workflow.add_node("buy",BuyNode(self,"add node:buyNode"))
        workflow.add_node("buy_query",BuyQueryNode(self,"add node:buyQueryNode"))
        workflow.add_node("agent",AgentNode(self,"add node:agentNode"))
        
        workflow.add_edge(START,"tag")
        workflow.add_conditional_edges("diet",self.router)
        workflow.add_conditional_edges("sport",self.router)
        workflow.add_conditional_edges("sign",self.router)
        workflow.add_conditional_edges("sign_query",self.router)
        workflow.add_conditional_edges("buy_query",self.router)
        workflow.add_conditional_edges("buy",self.router)
        workflow.add_edge("agent",END)
        workflow.add_conditional_edges("tag",self.router)
        
        graph = workflow.compile(self.memory)
        return graph 
               
    # def robot(self,state:State):
    #     #outputParser = self.graphLib.langchainLib.get_outputParser(Output)
    #     #prompt = self.graphLib.langchainLib.get_prompt("把以下文本翻译成中文",outputParser = outputParser)
        
    #     #prompt = self.graphLib.langchainLib.get_prompt("把以下文本翻译成中文")      
    #     llm = self.graphLib.langchainLib.get_llm(self.llm_key,self.llm_model)
    #     #chain =  prompt | llm.with_structured_output(Output)
    #     chain =  llm.bind_tools([self.add])
    #     messages = state["messages"]         
    #     res = chain.invoke(messages)
    #     #res = chain.invoke({"input":message.content})
    #     #print("Output(BaseModel)",res)
    #     if isinstance(res,BaseMessage):
    #         return {"messages":[res],"ask_human":False}
    #     elif res:        
    #         return {"messages":[AIMessage(content = str(res.targetText))],"ask_human":False}
    #     else:
    #         return {"messages":[AIMessage(content = "这个模型无法解析为Structed output")],"ask_human":False}
            