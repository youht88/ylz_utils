from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langchain.tools.retriever import create_retriever_tool

class RagSearchTool():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def get_tool(self,retriever,name,description):
        return create_retriever_tool(retriever,name,description)