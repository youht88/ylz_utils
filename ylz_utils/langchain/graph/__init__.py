from operator import itemgetter
from typing import Literal,List,Annotated
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,BaseMessage,ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import TypedDict
from langchain_core.tools import tool,Tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.state import CompiledStateGraph

from ylz_utils.file import FileLib
from ylz_utils.data import StringLib,Color
from ylz_utils.langchain.graph.common import State
from ylz_utils.langchain.graph.stand_graph import StandGraph
#from ylz_utils.langchain import LangchainLib 

class GraphLib():
    def __init__(self,langchainLib,db_conn_string=":memory:"):
        self.langchainLib = langchainLib
        #self.websearch_tool = self.set_websearch_tool(websearch_key)        
        #self.ragsearch_tool = langchainLib.get_ragsearch_tool(retriever)
        self.websearch_tool = None
        self.ragsearch_tool = None
        self.python_repl_tool = langchainLib.get_python_repl_tool()
        self.memory = SqliteSaver.from_conn_string(db_conn_string)

        self.llm_with_tools = None

        self.stand_graph = StandGraph(self)
    def set_dbname(self,dbname):
        # "checkpoint.sqlite"
        self.memory = SqliteSaver.from_conn_string(dbname)
    def set_ragsearch_tool(self,retriever):
        name = "rag_searcher"
        description = "一个有用的工具用来从本地知识库中获取信息。你总是利用这个工具优先从本地知识库中搜索有用的信息"
        self.ragsearch_tool = self.langchainLib.get_ragsearch_tool(retriever,name,description)
    def set_websearch_tool(self,websearch_key):
        self.websearch_tool = self.langchainLib.get_websearch_tool(websearch_key)
    
    def translate_to_en(self,state:State):
        return state
        content = state["messages"][-1].content
        prompt = self.langchainLib.get_prompt(
"""
将以下句子翻译成英文,注意不要遗漏任何信息，数字不是整数的话要保留所有整数和小数，不要翻译任何公司或人名
然后一步一步获得最终答案
""",
        use_chat=False)
        #llm = self.langchainLib.get_llm(model = "llama3-groq-70b-8192-tool-use-preview")
        #llm = self.langchainLib.get_llm(model = "llama3-70b-8192")
        llm = self.langchainLib.get_llm(key = "LLM.TOGETHER")
        chain = prompt | llm
        response = chain.invoke({"input":content})
        return {"messages":[response],"ask_human":False}
    
    
    def create_response(self, response: str, ai_message: AIMessage):
        return ToolMessage(
            content = response,
            tool_call_id = ai_message.tools_calls[0]["id"]
        )
    
    def human_node(self, state: State):
        new_messages = []
        if not isinstance(state["messages"][-1],ToolMessage):
            new_messages.append(
                self.create_response("No response from human.",state["messages"][-1])
            )
        return {
            "messages": new_messages,
            "ask_human": False
        }
    
    def select_next_node(self, state:State) -> Literal["human","tools","__end__"]:
        if state["ask_human"]:
            return "human"
        return tools_condition(state)
     
    def get_graph(self,llm_key=None,llm_model=None,key:Literal['stand_graph']='stand_graph',):
        if key=='other_graph':
            pass
        else:
            return self.stand_graph.get_graph(llm_key=llm_key,llm_model=llm_model)

    def graph_stream(self,graph,message,thread_id="default-default",system_message=None):    
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
                        print(f"{Color.LBLUE}AI:{Color.RESET}",msg_repr,
                              f'[model:{Color.LYELLOW}{message.response_metadata.get("model_name")}{Color.RESET},token:{Color.LYELLOW}{message.usage_metadata["total_tokens"]}{Color.RESET}]')
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
        if not as_node:
            graph.update_state(config = {"configurable":{"thread_id":thread_id}},values=values)

    def export_graph(self,graph:CompiledStateGraph):
        FileLib.writeFile("graph.png",graph.get_graph().draw_mermaid_png(),mode="wb")    
