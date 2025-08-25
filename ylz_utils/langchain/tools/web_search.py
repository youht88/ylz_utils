from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

import random
import re

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import GoogleSerperResults
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

from langchain_core.tools import Tool

from langchain.docstore.document import Document
from langchain_core.runnables import RunnableLambda

class WebSearchTool():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def get_tool_wrapper(self):
        search_config = self.config.get(f"SEARCH_TOOLS.TAVILY")        
        api_keys = search_config.get("API_KEYS")
        try:
            api_keys = self.langchainLib.split_keys(api_keys)
            api_key = random.choice(api_keys)
        except:
            raise Exception(f"请先设置TAVILY_API_KEYS环境变量") 
        search = TavilySearchAPIWrapper(tavily_api_key=api_key)
        tool = TavilySearchResults(api_wrapper=search,max_results=4)
        return tool

    def get_tool(self,key="TAVILY",rows=10):
        key = key.upper()
        if key == "DUCKDUCKGO":
            search = DuckDuckGoSearchAPIWrapper()
            tool  = DuckDuckGoSearchResults(api_wrapper=search,max_results=rows)
            #snippet,title,link: 
            pattern = "snippet: (.*?) title: (.*?) link: (.*?) snippet:"
            def __toDocument(text):
                pattern = r"\[snippet: (.*?) title: (.*?) link: (.*?)\]"
                matchs = re.findall(pattern,text)
                docs = [Document(match[0],metadata={"title":match[1],"link":match[2]}) for match in matchs]
                return docs
            tool_doc = tool | RunnableLambda(__toDocument, name="DDG2Document")
            #return tool_doc
            return tool 
        elif key == "TAVILY":
            # url,content,
            search_config = self.config.get(f"SEARCH_TOOLS.{key}")        
            api_keys = search_config.get("API_KEYS")
            try:
                api_keys = self.langchainLib.split_keys(api_keys)
                api_key = random.choice(api_keys)
            except:
                raise Exception(f"请先设置{key}_API_KEYS环境变量") 
            if api_key:
                print("TAVILY KEY:",api_key)
                try:
                    def __toDocument(doc_json):
                        docs = [Document(doc['content'],metadata={"link":doc['url']}) for doc in doc_json]
                        return docs
                    search = TavilySearchAPIWrapper(tavily_api_key=api_key)
                    tool = TavilySearchResults(api_wrapper=search,max_results=rows)
                    #tool_doc = tool | RunnableLambda(__toDocument, name="Tavily2Document")
                    #return tool_doc
                    return Tool(name=tool.name,func=tool,description=tool.description) 
                except:
                    raise Exception(f"请先设置{key}_API_KEYS环境变量") 
        elif key == "SERPAPI":
            # url,content,
            search_config = self.config.get(f"SEARCH_TOOLS.{key}")        
            api_keys = search_config.get("API_KEYS")
            try:
                api_keys = self.langchainLib.split_keys(api_keys)
                api_key = random.choice(api_keys)
            except:
                raise Exception(f"请先设置{key}_API_KEYS环境变量") 
            if api_key:
                try:
                    search = GoogleSerperAPIWrapper(serper_api_key=api_key,k = rows)
                    tool = GoogleSerperResults(api_wrapper=search)
                    #return tool
                    return Tool(name=tool.name,func=tool,description=tool.description)
                except Exception as e:
                    raise Exception(f"请先设置{key}_API_KEYS环境变量,{e}") 
        else:
            raise Exception(f"不支持{key}的搜索") 
