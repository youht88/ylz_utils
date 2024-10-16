from __future__ import annotations
from typing import TYPE_CHECKING,Optional

if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

import re
from abc import ABC,abstractmethod

from operator import itemgetter
from typing import Literal,List,Annotated
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage,RemoveMessage
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.tools import tool,Tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel,RunnableConfig
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.checkpoint.memory import MemorySaver
#from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolExecutor,ToolInvocation, tools_condition
from langgraph.graph.state import CompiledStateGraph

from ylz_utils.file import FileLib, IOLib
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
        self.memory = MemorySaver()
        #self.memory = SqliteSaver.from_conn_string(self.chat_db_name)
        #self.amemory = AsyncSqliteSaver.from_conn_string(self.chat_db_name)
        self.query_dbname = None
        self.tools=[]
        self.tools_executor = None
    
    class ConfigSchema(TypedDict):
        useSummary:bool
        thread_id:str
        user_id: Optional[str]
        system_message: Optional[str]
        llm_key: Optional[str]
        llm_model: Optional[str]
        

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
       
    def create_tool_message(self, response: str, ai_message: AIMessage):
        return ToolMessage(
            content = response,
            tool_call_id = ai_message.tools_calls[0]["id"]
        )
    def create_remove_messages(self,messages:BaseMessage):
        return [RemoveMessage(id=message["id"]) for message in messages ]
    
    def get_llm(self,llm_key=None,llm_model=None,config:Optional[RunnableConfig]=None):
        if config:
            llm_key = config.get('configurable',{}).get('llm_key')
            llm_model = config.get('configurable',{}).get('llm_model')
        return self.langchainLib.get_llm(llm_key,llm_model)
    def get_embedding(self,embedding_key=None,embedding_model=None):
        return self.langchainLib.get_embedding(embedding_key,embedding_model)
    
    @DeprecationWarning
    def  set_nodes_llm_config(self,nodes_llm_config:dict[str,dict[Literal["llm_key","llm_model"],str|None]]|tuple):
        # {
        #     "node1":{"llm_key":...,"llm_mode":...}
        #     "node2":{"llm_key":...,"llm_mode":...}
        # }
        # or
        # (llm_key,llm_model)
        if isinstance(nodes_llm_config,tuple):
            self.nodes_llm_config['default'] = {"llm_key":nodes_llm_config[0],"llm_model":nodes_llm_config[1]}
        else:
            self.nodes_llm_config = nodes_llm_config
    @DeprecationWarning
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
    @DeprecationWarning
    def set_thread(self,user_id="default",conversation_id="default"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.thread_id = f"{user_id}-{conversation_id}"

    def tools_execute(self,tools,state):
        if not tools_executor:
            raise Exception("please call set_tools_executor(tools) first!")
        messages = state["messages"]
        last_message = messages[-1]
        tool_messages = []
        for tool_call in last_message.tool_calls: 
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
            tool_messages.append(tool_message)
        return {"messages":tool_messages}
    def get_safe_response(self,responseMessage:BaseMessage):
        safeResponseMessage = responseMessage.model_copy()
        if isinstance(responseMessage,AIMessage):
            if len(responseMessage.tool_calls)>0 and responseMessage.content=="":
                print("!!!!!",responseMessage)
                names = list(set([tool_call["name"] for tool_call in responseMessage.tool_calls]))
                safeResponseMessage.content = f"我要使用`{','.join(names)}`等工具"
        return safeResponseMessage
    def get_class_instance_tools(self,classInstance)->list:
        '''获取类实例的所有函数，用于批量构成tools'''
        _exports = getattr(classInstance,"_exports")
        # 获取类的所有成员
        members = dir(classInstance)
        if _exports:
            members = [item for item in members if any([re.match(pattern,item) for pattern in _exports])]
            print("!!!!",members)
        # 筛选出函数
        methods = [getattr(classInstance,member) for member in members if callable(getattr(classInstance, member)) and not member.startswith("_")]
        #print(methods)
        return methods
    
    def summarize_conversation(self,state,config: RunnableConfig):
        llm_key = config["configurable"].get("llm_key")
        llm_model = config["configurable"].get("llm_model")
        if llm_key or llm_model:
            llm = self.langchainLib.get_llm(llm_key,llm_model)
        else:
            llm = self.get_node_llm()
        print("State keys:",state.keys(),state["summary"],len(state["messages"]))
        if len(state["messages"])>6:
            # First, we summarize the conversation
            summary = state.get("summary", "")
            if summary:
                # If a summary already exists, we use a different system prompt
                # to summarize it than if one didn't
                summary_message = (
                    f"This is summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account the new messages above:"
                )
            else:
                summary_message = "Create a summary of the conversation above:"
            messages =  state["messages"][:-1]+ [HumanMessage(content=summary_message)]
            response = llm.invoke(messages)
            print("response:",response)
            # We now need to delete messages that we no longer want to show up
            # I will delete all but the last two messages, but you can change this
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
            return {"summary": response.content, "messages": delete_messages }
        else:
            return {"messages":state["messages"]}
    @abstractmethod
    def get_graph(self) -> CompiledStateGraph:
        pass
    @abstractmethod
    def human_action(self,graph,config=None,thread_id=None) -> bool:
        return False
    
    def graph_test(self,graph:CompiledStateGraph,message,config=None,thread_id=None,stream_mode="values",subgraphs=False):
        if not config:
            if not thread_id:
                thread_id = self.thread_id
            config = {"configurable":{"thread_id":thread_id}}
        print("graph_test====>","config=",config)
        while True:
            if message=="/q":
                break
            self.graph_stream(graph,message,config,stream_mode=stream_mode,subgraphs=subgraphs)
            human_turn = self.human_action(graph,config)
            if human_turn:
                message = None
            else:
                message = IOLib.input_with_history(f"{StringLib.green('User: ')}") 

    def graph_stream(self,graph:CompiledStateGraph,message,config=None,thread_id=None,stream_mode="values",subgraphs=False): 
        if not config:
            if not thread_id:
                thread_id = self.thread_id
            config = {"configurable":{"thread_id":thread_id}}
        messages = []
        if message:
            messages.append(HumanMessage(content=message))
            input = {"messages":messages}          
        else:
            input = None
        events = graph.stream(  input,
                                config = config,
                                stream_mode = stream_mode,
                                subgraphs = subgraphs)
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
                        for tool_call in message.tool_calls:
                            print(f"{Color.LBLUE}AI:{Color.RESET}",f'使用{Color.GREEN}{tool_call["name"]}{Color.RESET},调用参数:{Color.GREEN}{tool_call["args"]}{Color.RESET}')
                    else:
                        response_metadata = message.response_metadata 
                        print(f"{Color.LBLUE}AI:{Color.RESET}",msg_repr,
                            f'[model:{Color.LYELLOW}{message.response_metadata.get("model_name")}{Color.RESET},token:{Color.LYELLOW}{message.response_metadata.get("token_usage",{}).get("total_tokens")}{Color.RESET}]')
                elif isinstance(message,ToolMessage):
                    print(f"    {Color.BLUE}Tool:{Color.RESET}",msg_repr)
                elif isinstance(message,HumanMessage):
                    print(f"{Color.BLUE}User:{Color.RESET} {msg_repr}")
                _printed.add(message.id)
                
    def graph_get_state_history(self,graph:CompiledStateGraph,config=None,thread_id=None):
        if not config:
            if not thread_id:
                thread_id = self.thread_id
            config = {"configurable":{"thread_id":thread_id}}
        state_history = graph.get_state_history(config = config )
        for state in state_history:
            print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
            print("-" * 80)
        return state_history
    
    def graph_get_state(self,graph:CompiledStateGraph,config=None,thread_id=None,subgraphs=False):
        if not config:
            if not thread_id:
                thread_id = self.thread_id
            config = {"configurable":{"thread_id":thread_id}}
        state = graph.get_state(config = config, subgraphs=subgraphs )
        return state
    
    def graph_update_state(self,graph:CompiledStateGraph,values,config=None,thread_id=None,as_node = None):
        if not config:
            if not thread_id:
                thread_id = self.thread_id
            config = {"configurable":{"thread_id":thread_id}}
        print("graph_update_state===>","as_node:",as_node,"thread_id:",thread_id,"values",values,"config:",config)
        graph.update_state(config = config,values=values,as_node=as_node)
    
    def graph_export(self,graph:CompiledStateGraph):
        FileLib.writeFile("graph.png",graph.get_graph(xray=1).draw_mermaid_png(),mode="wb")  

    