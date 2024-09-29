from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langgraph.prebuilt import chat_agent_executor
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import TavilySearchResults

class AgentLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
    def get_agent(self,llm,tools):
        return chat_agent_executor.create_function_calling_executor(llm,tools)
    def get_full_agent(self,*,llm=None,llm_key=None,llm_model=None):
        if not llm:
            llm = self.langchainLib.get_llm(llm_key,llm_model)    
        tool_tavily = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            # include_domains=[...],
            # exclude_domains=[...],
            # name="...",            # overwrite default tool name
            # description="...",     # overwrite default tool description
            # args_schema=...,       # overwrite default args_schema: BaseModel
        )
        tool_yahoo_finance_news = YahooFinanceNewsTool()
        tools = [tool_tavily]
        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        return agent_chain