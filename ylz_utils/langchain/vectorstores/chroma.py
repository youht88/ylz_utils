from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ylz_utils.langchain.vectorstores import VectorstoreLib

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from typing import List
from uuid import uuid4

class ChromaLib():
    def __init__(self,vectorstoreLib:VectorstoreLib):
        self.vectorstoreLib = vectorstoreLib
        self.config = vectorstoreLib.langchainLib.config
        self.host = self.config.get("CHROMA.HOST")
        self.port = self.config.get("CHROMA.PORT")
        self.dbname = self.config.get("CHROMA.DBNAME")
        if self.host and self.dbname :
            raise Exception("请指定host(SERVER模式)或dbname(本地模式)其中的一种模式")   
    def get_store(self,collection_name=None,embedding=None) -> Chroma:
        if not embedding:
            embedding = self.vectorstoreLib.langchainLib.get_embedding()
        if self.host:
            client = chromadb.HttpClient(host=self.host,port=self.port)
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                client=client,  # Where to save data locally, remove if not neccesary
            )
        else:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                persist_directory=self.dbname,  # Where to save data locally, remove if not neccesary
            )
        return vector_store
            
    def add_docs(self,vector_store: Chroma,docs,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(docs,batch,vector_store.add_documents)
        return all_ids

    def add_texts(self,vector_store: Chroma,texts,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(texts,batch,vector_store.add_texts)
        return all_ids
    
    def delete(self,vectorstore: Chroma,ids: List[str] | None = None):
        return vectorstore.delete(ids) 
          
    def search(self,query,vectorstore: Chroma,k=10,filter={}):
        return vectorstore.similarity_search(query,k=k,filter=filter)
    
    def search_with_score(self,query,vectorstore: Chroma,k=10,filter={}):
        return vectorstore.similarity_search_with_score(query,k=k,filter=filter)
