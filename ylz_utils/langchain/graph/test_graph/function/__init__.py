from ylz_utils.langchain.graph import GraphLib
from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode,tools_condition

from .node.agent import Agent

from .state import State
from .tools import Tools

class FunctionGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
        Tools(self)
        self.set_websearch_tool('tavily')
        #self.set_websearch_tool('duckduckgo')
        #self.set_websearch_tool('SERPAPI')
        self.tools.append(self.python_repl_tool)
        self.tools.append(self.websearch_tool)
        #print("!!!!!!!!!!!!",f"@{self.websearch_tool.api_wrapper.tavily_api_key.get_secret_value()}@")
    def get_graph(self):
        workflow = StateGraph(State)
        workflow.add_node("agent",Agent(self))
        workflow.add_node("tools",ToolNode(tools = self.tools))
        workflow.add_edge(START,"agent")
        workflow.add_conditional_edges("agent",tools_condition)
        workflow.add_edge("tools","agent")        
        graph = workflow.compile(self.memory)
        return graph

    def human_action(self, graph, config=None ,thread_id=None):
        return super().human_action(graph, config,thread_id)
    
