from __future__ import annotations
from typing import TYPE_CHECKING,Literal,TypedDict,Annotated

from ylz_utils.file import IOLib
if TYPE_CHECKING:
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

class StandGraph():
    user_id = 'default'
    conversation_id = 'default'
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib
        self.llm_with_tools = None
        self.score = Score(self)
        self.scoreNode = self.score.scoreNode
        self.scoreEdge = self.score.scoreEdge
        self.chatbotNode = ChatbotNode(self).chatbot
        self.humanNode = HumanNode(self).human_node

    def tool_node(self,state):
            messages = state["messages"]
            last_message = messages[-1]
            tool_call = last_message.tool_calls[0]
            action = ToolInvocation(
                tool = tool_call["name"],
                tool_input = tool_call["args"]
            )  
            response = self.tools_executor.invoke(action)
            tool_message = ToolMessage(
                content = str(response),
                name = action.tool,
                tool_call_id = tool_call["id"]
            )
            return {"messages":[tool_message]}
    
    def get_graph(self,llm_key=None,llm_model=None,user_id='default',conversation_id='default') -> CompiledStateGraph:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.llm = self.graphLib.langchainLib.get_llm(llm_key,llm_model)
        tools = [
            self.graphLib.python_repl_tool,
        ]
        if self.graphLib.websearch_tool:
            tools.append(self.graphLib.websearch_tool)
        if self.graphLib.ragsearch_tool:
            tools.append(self.graphLib.ragsearch_tool)
        self.tools_executor = self.graphLib.get_tools_executor(tools)
        # tools = [
        #        Tool(name="websearch",
        #             description = "the tool when you can not sure,please search from the internet",
        #             func = websearch
        #        ),
        #        Tool(name="python_repl",
        #             description="when you need to calculation, use python repl tool to execute code ,then return the result to AI.",
        #             func=self.python_repl_tool)
        #      ]
        
        self.llm_with_tools = self.llm.bind_tools(tools)      
        #self.llm_with_tools = llm.bind_tools(tools)      

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
        workflow.add_edge("humanNode","chatbotNode")

        graph = workflow.compile(checkpointer=self.graphLib.memory,
                                 interrupt_before= ["humanNode"])
        
        self.graphLib.regist_human_in_loop(graph,func = self.human_in_loop)
        return graph   

    def human_in_loop(self,graph,thread_id) -> bool:
        snapshot = self.graphLib.graph_get_state(graph,thread_id)
        if snapshot.next:
            next = snapshot.next[0]
            last_message = snapshot.values["messages"][-1]
            score = snapshot.values['score']
            print(f"AI: {score.request}")
            aiMessage = AIMessage(score.request)
            query = IOLib.input_with_history("User:")
            humanMessage = HumanMessage(query)
            snapshot.values["messages"].extend([aiMessage,humanMessage])
            self.graphLib.graph_update_state(graph,thread_id = thread_id,values=snapshot.values,as_node=next)
            return True
        return False     
 
