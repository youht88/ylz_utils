from __future__ import annotations
from typing import TYPE_CHECKING,Literal,TypedDict,Annotated

if TYPE_CHECKING:
    from ylz_utils.langchain.graph import GraphLib

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage

from .state import State
from .tag_node import TagNode
from .diet_node import DietNode
from .sport_node import SportNode
from .agent_node import AgentNode
from .router_edge import router

class LifeGraph():
    llm_key = None
    llm_model= None
    user_id = 'default'
    conversation_id = 'default'
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib
        self.tagNode = TagNode(self).tagNode
        self.dietNode = DietNode(self).dietNode
        self.sportNode = SportNode(self).sportNode
        self.agentNode = AgentNode(self).agentNode
        self.router = router

    def get_graph(self,llm_key=None,llm_model=None,user_id='default',conversation_id='default'):
        self.llm_key = llm_key
        self.llm_model = llm_model
        self.user_id = user_id
        self.conversation_id = conversation_id
        workflow = StateGraph(State)
        
        workflow.add_node("tag",self.tagNode)
        workflow.add_node("diet",self.dietNode)
        workflow.add_node("sport",self.sportNode)
        workflow.add_node("agent",self.agentNode)
        
        workflow.add_edge(START,"tag")
        workflow.add_edge("diet","agent")
        workflow.add_edge("sport","agent")
        workflow.add_edge("agent",END)
        workflow.add_conditional_edges("tag",self.router,{"agent":"agent","diet":"diet","sport":"sport","__end__":END})
        
        graph = workflow.compile(self.graphLib.memory)
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
            