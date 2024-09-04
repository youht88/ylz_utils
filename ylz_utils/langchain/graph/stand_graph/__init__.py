from ylz_utils.data import StringLib
from ylz_utils.file import IOLib

from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage
from langgraph.prebuilt import ToolNode

from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolExecutor,ToolInvocation, tools_condition

from .state import State,RequestAssistance
from .chatbot_node import ChatbotNode
from .human_node import HumanNode
from .score import Score
from .router_edge import router

class StandGraph(GraphLib):
    def __init__(self,langchainLib,db_conn_string=":memory:"):
        super().__init__(langchainLib,db_conn_string)
        self.llm_with_tools = None
        self.score = Score(self)
        self.scoreNode = self.score.scoreNode
        self.scoreEdge = self.score.scoreEdge
        self.chatbotNode = ChatbotNode(self).chatbot
        self.humanNode = HumanNode(self).human_node

    
    def get_graph(self,llm_key=None,llm_model=None,user_id='default',conversation_id='default') -> CompiledStateGraph:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.llm = self.set_llm(llm_key,llm_model)
        # tools = [
        #        Tool(name="websearch",
        #             description = "the tool when you can not sure,please search from the internet",
        #             func = websearch
        #        ),
        #        Tool(name="python_repl",
        #             description="when you need to calculation, use python repl tool to execute code ,then return the result to AI.",
        #             func=self.python_repl_tool)
        #      ]
        tools = self.set_tools()
        self.set_tools_executor(tools)
        
        self.llm_with_tools = self.llm.bind_tools(tools)           

        workflow = StateGraph(State)
        workflow.add_node("scoreNode",self.scoreNode)
        workflow.add_node("chatbotNode",self.chatbotNode)
        workflow.add_node("toolsNode",self.tool_node)
        workflow.add_node("humanNode", self.humanNode)
        
        workflow.add_conditional_edges("chatbotNode", router,
                                       {"human":"humanNode","tools":"toolsNode","__end__":"__end__"})
        
        workflow.add_edge(START, "scoreNode")
        workflow.add_conditional_edges("scoreNode",self.scoreEdge)
        workflow.add_edge("toolsNode","chatbotNode")
        workflow.add_edge("humanNode","scoreNode")

        graph = workflow.compile(checkpointer=self.memory,
                                 interrupt_before= ["humanNode"])
        
        return graph   

    def human_action(self,graph,thread_id) -> bool:
        snapshot = self.graph_get_state(graph,thread_id)
        if snapshot.next:
            next = snapshot.next[0]
            last_message = snapshot.values["messages"][-1]
            score = snapshot.values['score']
            print(f"    {StringLib.green('Question:')}: {score.request}")
            aiMessage = AIMessage(score.request)
            answer = IOLib.input_with_history(f"    {StringLib.green('Answer:')}")
            humanMessage = HumanMessage(answer)
            snapshot.values["messages"].extend([aiMessage,humanMessage])
            print(snapshot.values["messages"])
            self.graph_update_state(graph,thread_id = thread_id,values=snapshot.values,as_node=next)
            return True
        return False     
 
