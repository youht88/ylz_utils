
from langchain_community.document_loaders import DirectoryLoader

class DirLoader():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
        self.loader  = DirectoryLoader