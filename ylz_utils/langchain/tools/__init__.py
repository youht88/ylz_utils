from ylz_utils.langchain.tools.rag_search import RagSearchTool
from ylz_utils.langchain.tools.web_search import WebSearchTool


class ToolLib():
    def __init__(self,langchain):
        self.langchain = langchain
        self.web_search = WebSearchTool(langchain)
        self.rag_search = RagSearchTool(langchain)