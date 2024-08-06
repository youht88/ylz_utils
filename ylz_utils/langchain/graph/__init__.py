from typing import Literal,List,Annotated
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import TypedDict
from langchain_core.tools import tool,Tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition

from ylz_utils.file import FileLib
#from ylz_utils.langchain import LangchainLib
from langchain_community.tools.tavily_search import TavilySearchResults

tool_tavily = TavilySearchResults(max_results=2)
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

class State(TypedDict):
    messages: Annotated[list,add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str    

class GraphLib():
    def __init__(self,langchainLib,db_conn_string=":memory:"):
        self.langchainLib = langchainLib
        self.websearch_tool = langchainLib.get_websearch_tool
        self.ragsearch_tool = langchainLib.get_ragsearch_tool
        self.python_repl_tool = langchainLib.get_python_repl_tool
        self.memory = SqliteSaver.from_conn_string(db_conn_string)

        self.llm_with_tools = None
    def set_dbname(self,dbname):
        "checkpoint.sqlite"
        self.memory = SqliteSaver.from_conn_string(dbname)

    def chatbot(self,state: State):
        response = self.llm_with_tools.invoke(state["messages"])
        ask_human = False
        if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
            ask_human = True
        return {"messages":[response],"ask_human":ask_human}
    
    def router(self,state:MessagesState)-> Literal["tools","__end__"]:
            messages = state["messages"]
            tool_calls = messages[-1].tool_calls
            if tool_calls:
                return "tools"
            else:
                return END
    def create_response(self, response: str, ai_message: AIMessage):
        return ToolMessage(
            content = response,
            tool_call_id = ai_message.tools_calls[0]["id"]
        )
    
    def human_node(self, state: State):
        new_messages = []
        if not isinstance(state["messages"][-1],ToolMessage):
            new_messages.append(
                self.create_response("No response from human.",state["messages"])
            )
        return {
            "messages": new_messages,
            "ask_human": False
        }
    
    def select_next_node(self, state:State) -> Literal["human","tools","__end__"]:
        if state["ask_human"]:
            return "human"
        return tools_condition(state)
     
    def get_graph(self,llm_key=None,llm_model=None):
        llm = self.langchainLib.get_llm(llm_key,llm_model)
        tools = [
            tool_tavily,
            RequestAssistance
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

        workflow = StateGraph(State)
        workflow.add_node("chatbot",self.chatbot)
        workflow.add_node("tools",ToolNode(tools=[tool_tavily]))
        workflow.add_node("human", self.human_node)
        
        workflow.add_conditional_edges("chatbot", self.select_next_node,
                                       {"human":"human","tools":"tools","__end__":"__end__"})
        
        workflow.add_edge("tools","chatbot")
        workflow.add_edge("human","chatbot")
        workflow.add_edge(START, "chatbot")

        graph = workflow.compile(checkpointer=self.memory,
                                 interrupt_before= ["human"])

        return graph    

    def graph_stream(self,graph,message,thread_id="default-default"):    
        events = graph.stream({"messages":[HumanMessage(content = message)]},
                                config = {"configurable":{"thread_id":thread_id}},
                                stream_mode = "values")
        for event in events:
            if "messages" in event:
                event['messages'][-1].pretty_print()
    
    def graph_get_state_history(self,graph,config):
        state_history = graph.get_state_history(config)
        for state in state_history:
            print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
            print("-" * 80)
        return state_history
    
    def graph_get_state(self,graph,config):
        state = graph.get_state(config)
        print("Num Messages: ", state.values["messages"], "Next: ", state.next)
        return state
    
    def graph_update_state(self,graph,config,message):
        graph.update_state(config,{"messages":[message]})

    def export_graph(self,graph):
        FileLib.writeFile("graph.png",graph(xray=True).draw_mermaid_png(),mode="wb")    
