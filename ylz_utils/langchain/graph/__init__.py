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
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.user_id = 'default'
        self.conversation_id = 'default'
        self.thread_id='default-default'
        self.nodes_llm_config = {}
        self.websearch_tool = None
        self.ragsearch_tool = None
        self.python_repl_tool = langchainLib.get_python_repl_tool()
        self.chat_db_name = ":memory:"
        self.memory = SqliteSaver.from_conn_string(self.chat_db_name)
        self.query_dbname = None
        self.tools=[]
        self.tools_executor = None

    def set_chat_dbname(self,dbname):
        # "checkpoint.sqlite"
        self.chat_dbname = dbname
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

    def set_nodes_llm_config(self,nodes_llm_config:dict[str,dict[Literal["llm_key","llm_model"],str|None]]|tuple):
        # {
        #     "node1":{"llm_key":...,"llm_mode":...}
        #     "node2":{"llm_key":...,"llm_mode":...}
        # }
        if isinstance(nodes_llm_config,tuple):
            self.nodes_llm_config['default'] = {"llm_key":nodes_llm_config[0],"llm_model":nodes_llm_config[1]}
        else:
            self.nodes_llm_config = nodes_llm_config
    def get_node_llm(self,node_key=None):
        try:
            node_llm_config:dict = self.nodes_llm_config[node_key] 
            llm_key = node_llm_config.get("llm_key")
            llm_model = node_llm_config.get("llm_model")
            llm = self.langchainLib.get_llm(key=llm_key,model=llm_model)
        except:            
            default_llm_config = self.nodes_llm_config.get('default',{})
            if default_llm_config:
                llm = self.langchainLib.get_llm(key = default_llm_config.get('llm_key'), model = default_llm_config.get('llm_model'))
            else:
                llm = self.langchainLib.get_llm()
        finally:
            #print("llm=",llm.model_name,llm.openai_api_base)
            return llm
    def set_thread(self,user_id="default",conversation_id="default"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.thread_id = f"{user_id}-{conversation_id}"

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
    def get_graph(self) -> CompiledStateGraph:
        pass
    @abstractmethod
    def human_action(self,graph,thread_id=None):
        pass

    def graph_stream(self,graph:CompiledStateGraph,message,thread_id=None):    
        if not thread_id:
            thread_id = self.thread_id
        messages = []
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
                
    def graph_get_state_history(self,graph:CompiledStateGraph,thread_id=None):
        if not thread_id:
            thread_id = self.thread_id
        state_history = graph.get_state_history(config = {"configurable":{"thread_id":thread_id}} )
        for state in state_history:
            print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
            print("-" * 80)
        return state_history
    
    def graph_get_state(self,graph:CompiledStateGraph,thread_id=None):
        if not thread_id:
            thread_id = self.thread_id
        state = graph.get_state(config =  {"configurable":{"thread_id":thread_id}})
        return state
    
    def graph_update_state(self,graph:CompiledStateGraph,values,thread_id=None,as_node = None):
        if not thread_id:
            thread_id = self.thread_id
        print("as_node:",as_node,"thread_id:",thread_id,"values",values)
        graph.update_state(config = {"configurable":{"thread_id":thread_id}},values=values,as_node=as_node)
    
    def export_graph(self,graph:CompiledStateGraph):
        FileLib.writeFile("graph.png",graph.get_graph().draw_mermaid_png(),mode="wb")  

    