import logging
import inspect
import json
from datetime import datetime

from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage

from langchain.memory import ConversationBufferMemory

from langchain.docstore.document import Document

from langchain_community.tools.tavily_search.tool import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate
from langchain_core.prompt_values import StringPromptValue,ChatPromptValue

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda,Runnable
from operator import itemgetter

from langgraph.graph import START,END,MessageGraph,StateGraph
from langgraph.prebuilt import ToolNode

from ylz_utils.langchain.agents import AgentLib
from ylz_utils.langchain.llms import LLMLib
from ylz_utils.langchain.embeddings import EmbeddingLib
from ylz_utils.langchain.loaders import LoaderLib
from ylz_utils.langchain.prompts import PromptLib
from ylz_utils.langchain.output_parsers import OutputParserLib
from ylz_utils.langchain.spliters import SpliterLib
from ylz_utils.langchain.tools import ToolLib
from ylz_utils.langchain.vectorstores import VectorstoreLib

from ylz_utils.file import FileLib
from ylz_utils.config import Config
from ylz_utils.data import StringLib,Color

class LangchainLib():
    def __init__(self):
        self.config = Config.get()
        self.llmLib = LLMLib()
        self.embeddingLib = EmbeddingLib()
        self.promptLib = PromptLib()
        self.loaderLib = LoaderLib(self)
        self.spliterLib = SpliterLib()
        self.outputParserLib = OutputParserLib()
        self.vectorstoreLib = VectorstoreLib(self)
        self.toolLib = ToolLib(self)
        self.agentLib = AgentLib(self)
        #self.add_plugins()
        # 创建一个对话历史管理器
        self.memory = ConversationBufferMemory()

        self.get_chat = self.llmLib.get_chat
        self.get_llm = self.llmLib.get_llm
        self.get_embedding = self.embeddingLib.get_embedding
        self.get_prompt = self.promptLib.get_prompt
        self.get_outputParser = self.outputParserLib.get_outputParser
        self.get_websearch_tool = self.toolLib.web_search.get_tool
        self.get_ragsearch_tool = self.toolLib.rag_search.get_tool
        self.get_textsplitter = self.spliterLib.get_textsplitter
        self.split_markdown_docs = self.spliterLib.split_markdown_docs
        self.load_url_and_split_markdown = self.loaderLib.url.load_and_split_markdown
        self.get_agent = self.agentLib.get_agent
        
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
    