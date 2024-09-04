from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from abc import ABC,abstractmethod

from operator import itemgetter
from typing import Literal,List,Annotated
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import TypedDict
from langchain_core.tools import tool,Tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolExecutor,ToolInvocation, tools_condition
from langgraph.graph.state import CompiledStateGraph

from ylz_utils.file import FileLib
from ylz_utils.data import StringLib,Color

class GraphLib(ABC):
    def __init__(self,langchainLib:LangchainLib,db_conn_string=":memory:"):
        self.langchainLib = langchainLib
        self.node_llms = {}
        self.llm_key = None
        self.llm_model = None
        self.llm = None
        self.user_id = 'default'
        self.conversation_id = 'default'
        self.websearch_tool = None
        self.ragsearch_tool = None
        self.python_repl_tool = langchainLib.get_python_repl_tool()
        self.memory = SqliteSaver.from_conn_string(db_conn_string)
        self.query_dbname = None
        self.tools=[]
        self.tools_executor = None

    def set_llm(self,llm_key,llm_model):
        self.llm_key = llm_key
        self.llm_model = llm_model
        self.llm = self.langchainLib.get_llm(llm_key,llm_model)
        return self.llm

    def set_chat_db(self,dbname):
        # "checkpoint.sqlite"
        self.memory = SqliteSaver.from_conn_string(dbname)
    def set_query_dbname(self,dbname):
        self.query_dbname = dbname
    def set_ragsearch_tool(self,retriever):
        name = "rag_searcher"
        description = "一个有用的工具用来从本地知识库中获取信息。你总是利用这个工具优先从本地知识库中搜索有用的信息"
        self.ragsearch_tool = self.langchainLib.get_ragsearch_tool(retriever,name,description)
    def set_websearch_tool(self,websearch_key):
        self.websearch_tool = self.langchainLib.get_websearch_tool(websearch_key)
    def set_tools(self,tools=None):
        if not tools:
            tools = [
                self.python_repl_tool,
            ]
            if self.websearch_tool:
                tools.append(self.websearch_tool)
            if self.ragsearch_tool:
                tools.append(self.ragsearch_tool)
        self.tools = tools
        return self.tools    
    def set_tools_executor(self,tools):
        if not isinstance(tools,(list,tuple)):
            tools = [tools]
        self.tools_executor = ToolExecutor(tools) 
        return self.tools_executor
       
    def create_response(self, response: str, ai_message: AIMessage):
        return ToolMessage(
            content = response,
            tool_call_id = ai_message.tools_calls[0]["id"]
        )

    def set_node_llms(self,node_llms):
        self.node_llms = node_llms

    def get_node_llm(self,node_key):
        try:
            node_llm:dict = self.node_llms[node_key] 
            llm_key = node_llm.get("llm_key")
            llm_model = node_llm.get("llm_model")
            return self.langchainLib.get_llm(key=llm_key,model=llm_model)
        except:            
            return self.langchainLib.get_llm(key = self.llm_key, model = self.llm_model)
        
    def tool_node(self,state):
        if not self.tools_executor:
            raise Exception("please call set_tools_executor(tools) first!")
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
        
    @abstractmethod
    def get_graph(self,llm_key=None,llm_model=None,user_id='default',conversation_id='default') -> CompiledStateGraph:
        pass
    # def get_graph(self,graph_key:Literal['stand','life','engineer','db','selfrag']='stand',llm_key=None,llm_model=None,user_id='default',conversation_id='default'):
    #     if graph_key=='life':
    #         return self.life_graph.get_graph(llm_key,llm_model,user_id,conversation_id)
    #     elif graph_key=='engineer':
    #         return self.engineer_graph.get_graph(llm_key,llm_model,user_id,conversation_id)
    #     elif graph_key=='db':
    #         self.db_graph.set_db(f"sqlite:///{self.query_dbname}")
    #         #self.db_graph.set_db("sqlite:///person.db")
    #         self.db_graph.set_llm(llm_key,llm_model)
    #         return self.db_graph.get_graph(llm_key,llm_model,user_id,conversation_id)
    #     elif graph_key=='selfrag':
    #         self.self_rag_graph.set_retriever()
    #         return self.self_rag_graph.get_graph(llm_key,llm_model,user_id,conversation_id)
    #     else:
    #         return self.stand_graph.get_graph(llm_key,llm_model,user_id,conversation_id)
    @abstractmethod
    def human_action(self,graph,thread_id):
        pass

    def graph_stream(self,graph:CompiledStateGraph,message,thread_id="default-default",system_message=None):    
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        if message:
            messages.append(HumanMessage(content=message))
            values = {"messages":messages}          
        else:
            values = None
        events = graph.stream( values,
                                config = {"configurable":{"thread_id":thread_id}},
                                stream_mode = "values")
        _print_set = set()
        for event in events:
            self._print_event(event,_print_set)
    
    def _print_event(self, event: dict, _printed: set, max_length=1500):
        current_state = event.get("dialog_state")
        if current_state:
            print("Currently in: ", current_state[-1])
        message = event.get("messages")
        if message:
            if isinstance(message, list):
                message = message[-1]
            if message.id not in _printed:
                msg_repr = message.content
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                if isinstance(message,AIMessage):
                    if message.tool_calls:
                        print(f"{Color.LBLUE}AI:{Color.RESET}",f'使用{Color.GREEN}{message.tool_calls[0]["name"]}{Color.RESET},调用参数:{Color.GREEN}{message.tool_calls[0]["args"]}{Color.RESET}')
                    else:
                        response_metadata = message.response_metadata 
                        print(f"{Color.LBLUE}AI:{Color.RESET}",msg_repr,
                            f'[model:{Color.LYELLOW}{message.response_metadata.get("model_name")}{Color.RESET},token:{Color.LYELLOW}{message.response_metadata.get("token_usage",{}).get("total_tokens")}{Color.RESET}]')
                elif isinstance(message,ToolMessage):
                    print(f"    {Color.BLUE}Tool:{Color.RESET}",msg_repr)
                elif isinstance(message,HumanMessage):
                    print(f"{Color.BLUE}User:{Color.RESET} {msg_repr}")
                _printed.add(message.id)
                
    def graph_get_state_history(self,graph:CompiledStateGraph,thread_id="default-default"):
        state_history = graph.get_state_history(config = {"configurable":{"thread_id":thread_id}} )
        for state in state_history:
            print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
            print("-" * 80)
        return state_history
    
    def graph_get_state(self,graph:CompiledStateGraph,thread_id="default-default"):
        state = graph.get_state(config =  {"configurable":{"thread_id":thread_id}})
        return state
    
    def graph_update_state(self,graph:CompiledStateGraph,thread_id,values,as_node = None):
        print("as_node:",as_node,"thread_id:",thread_id,"values",values)
        graph.update_state(config = {"configurable":{"thread_id":thread_id}},values=values,as_node=as_node)
    
    def export_graph(self,graph:CompiledStateGraph):
        FileLib.writeFile("graph.png",graph.get_graph().draw_mermaid_png(),mode="wb")  

    