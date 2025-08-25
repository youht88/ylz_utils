from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_community.document_loaders import DirectoryLoader

from ylz_utils.langchain import LangchainLib
from ylz_utils.crypto import HashLib

class DocumentLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
    
    @abstractmethod   
    def loader(self, file_name: str):
        raise NotImplementedError("loader method must be implemented")

    def dirloader(self, dir_path: str, glob:str,show_progress=True):
        return DirectoryLoader(dir_path,glob=glob,show_progress=show_progress,loader_cls=self.loader)     

    def spliter(self, chunk_size=1000,chunk_overlap=0):
        spliter:RecursiveCharacterTextSplitter = self.langchainLib.get_textsplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap)
        return spliter

    def load(self, file_name: str, metadata=None):
        loader = self.loader(file_name)
        docs:List[Document] = loader.load()
        if docs:
            file_hash = HashLib.sha256(docs[0].page_content)
            docs[0].metadata.update({"source_sha256":file_hash})
            if metadata:
                docs[0].metadata.update(metadata)
        return docs 

    def dirload(self, dir_path: str, glob:str,show_progress=True,metadata=None):
        loader = self.dirloader(dir_path,glob,show_progress)
        docs:List[Document] = loader.load()
        print(f"dir load {len(docs)} docs")
        for doc in  docs:
            file_hash = HashLib.sha256(doc.page_content)
            doc.metadata.update({"source_sha256":file_hash})
            if metadata:
                doc.metadata.update(metadata)   
        return docs
    
    def load_and_split(self, file_name, metadata=None,chunk_size=1000,chunk_overlap=0):
        docs = self.load(file_name,metadata)
        spliter = self.spliter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)        
        splited_docs = spliter.split_documents(docs)
        return splited_docs
        
    def dirload_and_split(self, dir_path, glob, metadata=None,chunk_size=1000,chunk_overlap=0):
        docs = self.dirload(dir_path, glob,show_progress=True,metadata=metadata)
        spliter:RecursiveCharacterTextSplitter = self.langchainLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        splited_docs = spliter.split_documents(docs)
        return splited_docs
        