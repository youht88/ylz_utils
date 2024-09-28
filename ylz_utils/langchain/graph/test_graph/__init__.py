from ylz_utils.langchain.graph import GraphLib

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START,END,StateGraph,MessagesState
from .function import FunctionGraph
from .summary import SummaryGraph

class TestGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
    def get_graph(self) -> CompiledStateGraph:
        # workflow = StateGraph(MessagesState)
        # #workflow.add_node("testAgent",FunctionGraph(self.langchainLib).get_graph())
        # workflow.add_node("testAgent",SummaryGraph(self.langchainLib).get_graph())
        # workflow.add_edge(START,"testAgent")
        # workflow.add_edge("testAgent",END)
        # graph = workflow.compile(self.memory)
        # return graph
        return SummaryGraph(self.langchainLib).get_graph()
    def human_action(self, graph, config=None, thread_id=None) -> bool:
        return super().human_action(graph, config, thread_id)
    
