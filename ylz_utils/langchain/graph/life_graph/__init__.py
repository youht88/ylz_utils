
from ylz_utils.langchain.graph import GraphLib

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage

from .state import State
from .tag_node import TagNode
from .diet_node import DietNode
from .sport_node import SportNode
from .sign_node import SignNode
from .sign_query_node import SignQueryNode
from .buy_node import BuyNode
from .agent_node import AgentNode
from .router_edge import router

class LifeGraph(GraphLib):
    def __init__(self,langchainLib,db_conn_string=":memory:"):
        super().__init__(langchainLib,db_conn_string)
        self.tagNode = TagNode(self).tagNode
        self.dietNode = DietNode(self).dietNode
        self.sportNode = SportNode(self).sportNode
        self.signNode = SignNode(self).signNode
        self.signQueryNode = SignQueryNode(self).signQueryNode
        self.buyNode = BuyNode(self).buyNode
        self.agentNode = AgentNode(self).agentNode
        self.router = router
    
    def human_action(self, graph, thread_id):
        return super().human_action(graph, thread_id)
    
    def get_graph(self,llm_key=None,llm_model=None,user_id='default',conversation_id='default'):
        self.llm_key = llm_key
        self.llm_model = llm_model
        self.user_id = user_id
        self.conversation_id = conversation_id
        workflow = StateGraph(State)
        
        workflow.add_node("tag",self.tagNode)
        workflow.add_node("diet",self.dietNode)
        workflow.add_node("sport",self.sportNode)
        workflow.add_node("sign",self.signNode)
        workflow.add_node("sign_query",self.signQueryNode)
        workflow.add_node("buy",self.buyNode)
        workflow.add_node("agent",self.agentNode)
        
        workflow.add_edge(START,"tag")
        workflow.add_conditional_edges("diet",self.router)
        workflow.add_conditional_edges("sport",self.router)
        workflow.add_conditional_edges("sign",self.router)
        workflow.add_conditional_edges("sign_query",self.router)
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
            