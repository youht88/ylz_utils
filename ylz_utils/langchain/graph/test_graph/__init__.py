from ylz_utils.langchain.graph import GraphLib
from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_core.messages import AIMessage
class TestGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
    
    def get_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("nodeA",NodeA("hello",self))
        workflow.add_node("nodeB",NodeB("3+3=?",self))
        workflow.add_edge(START,"nodeA")
        workflow.add_edge("nodeA","nodeB")
        workflow.add_edge("nodeB",END)        
        graph = workflow.compile(self.memory)
        return graph

    def human_action(self, graph, thread_id=None):
        return super().human_action(graph, thread_id)
    
class NodeA():
    def __init__(self,msg,graphLib:GraphLib=None):
        self.llm=None
        if graphLib:
            self.llm = graphLib.get_node_llm()
        self.msg = msg
    def __call__(self,state):
        if self.llm:
            res = self.llm.invoke(self.msg)
            return {"messages":[res]}
        return {"messages":[AIMessage(content=self.msg)]}

class NodeB():
    def __init__(self,msg,graphLib:GraphLib=None):
        self.llm=None
        if graphLib:
            self.llm = graphLib.get_node_llm()
        self.msg = msg
        self.tools = [self.add]
    def add(self,a:int , b:int)->int:
        '''将两个整数相加'''
        return a+b
    def __call__(self,state):
        if self.llm:
            llm_bind = self.llm.bind_tools(self.tools)
            res = llm_bind.invoke(self.msg)
            print("???",res)
            return {"messages":[res]}
        return {"messages":[AIMessage(content=self.msg)]}
    