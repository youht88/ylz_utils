from langchain.tools.retriever import create_retriever_tool

class RagSearchTool():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def get_tool(self,retriever,name,description):
        return create_retriever_tool(retriever,name,description)