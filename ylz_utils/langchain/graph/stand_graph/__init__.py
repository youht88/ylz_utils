from typing import Literal
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
from .score_node import ScoreNode

class StandGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        
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

    def chatbot_router(self,state:State)-> Literal["tools","__end__"]:
        messages = state["messages"]
        tool_calls = messages[-1].tool_calls
        if tool_calls:
            return "tools"
        else:
            return END

    def score_router(self,state:State):
        score = state["score"]
        print("score_router.score=",score)
        if score.score <= 1:
            return "chatbotNode"
        else:
            return "humanNode"


    def get_graph(self) -> CompiledStateGraph:
        print("--> create graph:standGraph")
        workflow = StateGraph(State)
        workflow.add_node("scoreNode",ScoreNode(self,"add node:scoreNode"))
        workflow.add_node("chatbotNode",ChatbotNode(self,"add node:chatbotNode"))
        workflow.add_node("toolsNode",self.tool_node)
        workflow.add_node("humanNode", HumanNode(self,"add node:humanNode"))
        
        workflow.add_conditional_edges("chatbotNode", self.chatbot_router,
                                       {"human":"humanNode","tools":"toolsNode","__end__":"__end__"})
        
        workflow.add_edge(START, "scoreNode")
        workflow.add_conditional_edges("scoreNode",self.score_router)
        workflow.add_edge("toolsNode","chatbotNode")
        workflow.add_edge("humanNode","scoreNode")

        graph = workflow.compile(checkpointer=self.memory,
                                 interrupt_before= ["humanNode"])
        
        return graph   

    def human_action(self,graph,thread_id=None) -> bool:
        if not thread_id:
            thread_id = self.thread_id
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
 
