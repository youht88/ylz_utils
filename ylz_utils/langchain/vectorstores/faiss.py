from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from ylz_utils.langchain import LangchainLib
from ylz_utils.langchain.vectorstores import VectorstoreLib
import logging

try:
    import faiss
except:
    logging.warning(f"use `pip install faiss` ,otherwise you can't use provider by FaissLib")

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from typing import List
from uuid import uuid4

class FaissLib(VectorstoreLib):
    def __init__(self,langchainLib:LangchainLib,db_file=None):
        super().__init__(langchainLib)
        self.db_file = db_file or self.config.get("VECTORSTORE.FAISS.DB_FILE")
        self.collection_name = "index"
    def get_store(self,collection_name=None,embedding=None) -> FAISS:
        if collection_name:
            self.collection_name = collection_name
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        print("????",self.collection_name,self.db_file,embedding)
        index = faiss.IndexFlatL2(len(embedding.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store
            
    def save(self,  vectorstore: FAISS,*,db_file:str=None,collection_name:str = None):
        if not db_file:
            db_file = self.db_file
        if not collection_name:
            collection_name = self.collection_name
        vectorstore.save_local(db_file,collection_name)

    def load(self, db_file:str=None , *, embedding=None, collection_name:str = None) -> FAISS:
        if not db_file:
            db_file = self.db_file
        if not collection_name:
            collection_name = self.collection_name
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        vectorestore = FAISS.load_local(db_file, embeddings=embedding, index_name=collection_name, allow_dangerous_deserialization=True)
        return vectorestore
    
