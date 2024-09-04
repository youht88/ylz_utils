import logging
import inspect
import json
import os
from datetime import datetime

from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage



from langchain.docstore.document import Document

from langchain_community.tools.tavily_search.tool import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate
from langchain_core.prompt_values import StringPromptValue,ChatPromptValue

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda,Runnable
from operator import itemgetter

from langgraph.graph import START,END,MessageGraph,StateGraph
from langgraph.prebuilt import ToolNode

from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.langchain.agents import AgentLib
from ylz_utils.langchain.flux import FluxLib
from ylz_utils.langchain.graph import GraphLib
from ylz_utils.langchain.llms import LLMLib
from ylz_utils.langchain.embeddings import EmbeddingLib
from ylz_utils.langchain.documents import DocumentLib
from ylz_utils.langchain.memory import MemoryLib
from ylz_utils.langchain.prompts import PromptLib
from ylz_utils.langchain.output_parsers import OutputParserLib
from ylz_utils.langchain.splitters import SplitterLib
from ylz_utils.langchain.tools import ToolLib
from ylz_utils.langchain.tts import TTSLib
from ylz_utils.langchain.vectorstores import VectorstoreLib

from ylz_utils.file import FileLib
from ylz_utils.config import Config
from ylz_utils.data import StringLib,Color

class LangchainLib():
    neo4jLib =None
    def __init__(self,trace=True):    
        self.config = Config.get()
        os.environ["LANGCHAIN_PROJECT"] = Config.project_name
        langsmith_trace = os.environ.get("LANGSMITH_TRACING_V2")
        if not langsmith_trace: 
            if trace:
                os.environ["LANGSMITH_TRACING_V2"] = "true"
            else:
                trace = False
        else:
            trace = True        
        if trace:        
            langsmith_api_keys = self.config.get('LANGSMITH.API_KEYS')
            langsmith_api_keys = self.split_keys(langsmith_api_keys)
            if langsmith_api_keys:
                os.environ["LANGSMITH_API_KEY"]=langsmith_api_keys[0]
        print("smith_tracing=",os.environ.get("LANGSMITH_TRACING_V2"))
        print("smith_api_key=",os.environ.get("LANGSMITH_API_KEY"))
        print("project_name=",os.environ["LANGCHAIN_PROJECT"])
        
        self.memoryLib = MemoryLib(self)

        self.llmLib = LLMLib(self)
        self.get_chat = self.llmLib.get_chat
        self.get_llm = self.llmLib.get_llm

        self.embeddingLib = EmbeddingLib(self)
        self.get_embedding = self.embeddingLib.get_embedding

        self.promptLib = PromptLib()
        self.get_prompt = self.promptLib.get_prompt

        self.outputParserLib = OutputParserLib()
        self.get_outputParser = self.outputParserLib.get_outputParser
        
        self.vectorstoreLib = VectorstoreLib(self)
        
        self.toolLib = ToolLib(self)
        self.get_websearch_tool = self.toolLib.web_search.get_tool
        self.get_ragsearch_tool = self.toolLib.rag_search.get_tool
        self.get_python_repl_tool = self.toolLib.python_repl.get_tool

        self.splitterLib = SplitterLib()
        self.get_textsplitter = self.splitterLib.get_textsplitter
        self.split_markdown_docs = self.splitterLib.split_markdown_docs

        self.documentLib = DocumentLib(self)
        self.load_url_and_split_markdown = self.documentLib.url.load_and_split_markdown

        self.agentLib = AgentLib(self)
        self.get_agent = self.agentLib.get_agent
        
        self.ttsLib = TTSLib(self)
        self.fluxLib = FluxLib(self)

    def init_neo4j(self,neo4jLib:Neo4jLib):
        self.neo4jLib = neo4jLib     
    
    def add_plugins(self,debug=False):
        plugins = [{"class":ChatOpenAI,"func":ChatOpenAI.invoke},
                   {"class":ChatOpenAI,"func":ChatOpenAI.ainvoke},
                   {"class":ChatOpenAI,"func":ChatOpenAI.stream},
                   {"class":ChatOpenAI,"func":ChatOpenAI.astream},
                   {"class":TavilySearchResults,"func":TavilySearchResults.invoke}
        ]
        def get_wrapper(func):
            logging.info(f"增加{func.__qualname__}的插件!!")
            def sync_wrapper(self, *args,**kwargs):
                if isinstance(args[0],StringPromptValue):
                    text = args[0].text
                    args[0].text = f"现在是北京时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}\n{text}"
                elif isinstance(args[0],ChatPromptValue):
                    chatPromptValue = args[0]
                    chatPromptValue.messages.insert(0,SystemMessage(f"现在是北京时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}"))
                elif isinstance(args[0],str):
                    new_args = list(args).copy()
                    text = new_args[0]
                    new_args[0] = f"现在是北京时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}\n{text}"                    
                    #print("?"*10,new_args,"现在仍使用原args")
                if debug:
                    StringLib.logging_in_box("提示词-->"+str(args))
                return func(self, *args, **kwargs)
            async def async_wrapper(self, *args,**kwargs):
                if isinstance(args[0],StringPromptValue):
                    text = args[0].text
                    args[0].text = f"现在是北京时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}\n{text}"
                elif isinstance(args[0],ChatPromptValue):
                    chatPromptValue = args[0]
                    chatPromptValue.messages.insert(0,SystemMessage(f"现在是北京时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}"))
                if debug:
                    StringLib.logging_in_box("提示词-->"+str(args))
                return await func(self, *args, **kwargs)
            if  inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        for item in plugins:
            func = item["func"]
            func_name = func.__name__
            cls = item["class"]
            setattr(cls,func_name,get_wrapper(func))
    def split_keys(self,keys_str: str | list[str]):
        keys = []

        if keys_str:
            new_keys_str = keys_str
            if isinstance(new_keys_str,list):
                new_keys_str = filter(lambda x:x,new_keys_str)
                new_keys_str = ",".join(new_keys_str)
            keys = list(set(filter(lambda x:x,new_keys_str.split(","))))
        return keys        
