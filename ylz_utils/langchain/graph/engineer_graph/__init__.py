from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph import GraphLib

from typing import Literal

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from .check import check
from .critique import critique
from .draft import draft_answer
from .gather_requirements import gather_requirements
from .state import AgentState, OutputState, GraphConfig

class EngineerGraph():
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib
        # self.node_llms = {
        #     "draft_answer": {"llm_key": "LLM.GROQ","llm_model":None},
        #     "gather_requirements": {"llm_key": "LLM.GROQ","llm_model":None},
        #     "critique": {"llm_key": "LLM.GROQ","llm_model":None},
        # }
        self.node_llms = None
        self.llm_key = None
        self.llm_model = None
    def route_critique(self,state: AgentState) -> Literal["draft_answer", END]:
        if state['accepted']:
            return END
        else:
            return "draft_answer"
    def route_check(self, state: AgentState) -> Literal["critique", "draft_answer"]:
        if isinstance(state['messages'][-1], AIMessage):
            return "critique"
        else:
            return "draft_answer"
    def route_start(self, state: AgentState) -> Literal["draft_answer", "gather_requirements"]:
        if state.get('requirements'):
            return "draft_answer"
        else:
            return "gather_requirements"
    def route_gather(self, state: AgentState) -> Literal["draft_answer", END]:
        if state.get('requirements'):
            return "draft_answer"
        else:
            return END
    def set_node_llms(self,node_llms):
        self.node_llms = node_llms

    def get_node_llm(self,node_key):
        try:
            node_llm = self.node_llms[node_key] 
            llm_key = node_llm.get("llm_key")
            llm_model = node_llm.get("llm_model")
            return self.graphLib.langchainLib.get_llm(key=llm_key,model=llm_model)
        except:            
            return self.graphLib.langchainLib.get_llm(key = self.llm_key, model = self.llm_model)

    def get_graph(self,llm_key=None,llm_model=None) -> CompiledStateGraph:
        # Define a new graph
        self.llm_key = llm_key
        self.llm_model = llm_model
        workflow = StateGraph(AgentState, input=MessagesState, output=OutputState, config_schema=GraphConfig)
        workflow.add_node(draft_answer)
        workflow.add_node(gather_requirements)
        workflow.add_node(critique)
        workflow.add_node(check)
        workflow.set_conditional_entry_point(self.route_start)
        workflow.add_conditional_edges("gather_requirements", self.route_gather)
        workflow.add_edge("draft_answer", "check")
        workflow.add_conditional_edges("check", self.route_check)
        workflow.add_conditional_edges("critique", self.route_critique)
        graph = workflow.compile(checkpointer=self.graphLib.memory)
        return graph 
    



