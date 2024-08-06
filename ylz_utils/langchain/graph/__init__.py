from typing import Literal,List

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool,Tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode

from ylz_utils.file import FileLib
#from ylz_utils.langchain import LangchainLib
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
# tools = [tool]
# tool.invoke("What's a 'node' in LangGraph?")

def websearch(self,query):
    '''
        use TAVILY to search from the internet
        arguments:
            self: the first args must be self
            query: the query which should be search from the internet . string it must be single-str format 
    '''
    print(query)
    if isinstance(query,list):
        query = query[0]
    query = query
    #res = websearch_tool.invoke(query)
    res = "中国，北京"
    if res:
        return res
    else:
        return "I don't search result from the internet."
class GraphLib():
    def __init__(self,langchainLib,db_conn_string=":memory:"):
        self.langchainLib = langchainLib
        self.websearch_tool = langchainLib.get_websearch_tool
        self.ragsearch_tool = langchainLib.get_ragsearch_tool
        self.python_repl_tool = langchainLib.get_python_repl_tool
        self.memory = SqliteSaver.from_conn_string(db_conn_string)

        self.llm_with_tools = None
    def set_dbname(self,dbname):
        "checkpoints.sqlite"
        self.memory = SqliteSaver.from_conn_string(dbname)
    def call_llm(self,state:MessagesState):
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages":[response]}
    
    def router(self,state:MessagesState)-> Literal["tools","__end__"]:
            messages = state["messages"]
            tool_calls = messages[-1].tool_calls
            if tool_calls:
                return "tools"
            else:
                return END
    
    def get_graph(self,llm_key=None,llm_model=None):
        llm = self.langchainLib.get_llm(llm_key,llm_model)
        tools = [
            Tool(name = "websearch",
                 description = "the tool when you can not sure,please search from the internet",    
                 func = tool
            )
        ]
        # tools = [
        #        Tool(name="websearch",
        #             description = "the tool when you can not sure,please search from the internet",
        #             func = websearch
        #        ),
        #        Tool(name="python_repl",
        #             description="when you need to calculation, use python repl tool to execute code ,then return the result to AI.",
        #             func=self.python_repl_tool)
        #      ]
        self.llm_with_tools = llm.bind_tools(tools)      

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent",self.call_llm)
        tool_node = ToolNode(tools=tools)
        workflow.add_node("tools",tool_node)
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges("agent",self.router)
        workflow.add_edge("tools","agent")

        graph = workflow.compile(checkpointer=self.memory)

        return graph    

    def graph_invoke(self,graph,message,thread_id="default-default"):    
        final_state = graph.invoke({"messages":[HumanMessage(content = message)]},
                                config = {"configurable":{"thread_id":thread_id}})
        print("*"*80)
        print(final_state)
        print("*"*80)
        return final_state['messages'][-1].content
    
    def export_graph(self,graph):
        FileLib.writeFile("graph.png",graph(xray=True).draw_mermaid_png(),mode="wb")    
