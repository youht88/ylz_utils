from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from ylz_utils.langchain.tools.python_repl import PythonREPLTool
from ylz_utils.langchain.tools.rag_search import RagSearchTool
from ylz_utils.langchain.tools.web_search import WebSearchTool
from ylz_utils.langchain.tools.wolfram_alpha import WolframAlphaTool


class ToolLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.web_search = WebSearchTool(langchainLib)
        self.rag_search = RagSearchTool(langchainLib)
        self.python_repl = PythonREPLTool(langchainLib)
        self.wolfram_alpha = WolframAlphaTool(langchainLib)