
from typing import Literal
from ylz_utils.langchain.graph import GraphLib

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage


from .state import State, Tag
from .tag_node import TagNode
from .life_node import LifeNode
from .life_query_node import LifeQueryNode
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
    
    def router(self,state:State)->Literal["life","life_query","agent","__end__"]:
        tag:Tag = state["life_tag"]
        print("tag=======>",tag)
        if tag.is_question:
            for item in tag.subTags:
                match item.type:
                    case "diet":
                        return "life_query"
                    case "sport":
                        return "life_query"
                    case "sign":
                        return "life_query"
                    case "buy":
                        return "life_query"
                    case _:
                        return "agent"
            return "agent"
        match tag.action:
            case "record":
                for item in tag.subTags:
                    match item.type:
                        case "diet":
                            return "life"
                        case "sport":
                            return "life"
                        case "sign":
                            return "life"
                        case "buy":
                            return "life"
                        case _:
                            return "agent"
                return "agent"
            case _:
                return "agent"
    
    def get_graph(self):
        print("---> create graph:lifeGraph")
        workflow = StateGraph(State)
        
        workflow.add_node("tag",TagNode(self,"add node:tagNode"))
        workflow.add_node("life",LifeNode(self,"add node:lifeNode"))
        workflow.add_node("life_query",LifeQueryNode(self,"add node:lifeQueryNode"))
        workflow.add_node("agent",AgentNode(self,"add node:agentNode"))
        
        workflow.add_edge(START,"tag")
        workflow.add_conditional_edges("life",self.router)
        workflow.add_conditional_edges("life_query",self.router)
        workflow.add_edge("agent",END)
        workflow.add_conditional_edges("tag",self.router)
        
        graph = workflow.compile(self.memory)
        return graph 