from ylz_utils.langchain.tools.python_repl import PythonREPLTool
from ylz_utils.langchain.tools.rag_search import RagSearchTool
from ylz_utils.langchain.tools.web_search import WebSearchTool
from ylz_utils.langchain.tools.wolfram_alpha import WolframAlphaTool


class ToolLib():
    def __init__(self,langchain):
        self.langchain = langchain
        self.web_search = WebSearchTool(langchain)
        self.rag_search = RagSearchTool(langchain)
        self.python_repl = PythonREPLTool(langchain)
        self.wolfram_alpha = WolframAlphaTool(langchain)