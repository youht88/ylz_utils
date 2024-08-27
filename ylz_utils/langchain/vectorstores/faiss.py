from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ylz_utils.langchain.vectorstores import VectorstoreLib

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from typing import List
from uuid import uuid4

class FaissLib():
    def __init__(self,vectorstoreLib:VectorstoreLib):
        self.vectorstoreLib = vectorstoreLib
        self.config = self.vectorstoreLib.langchainLib.config
        self.db_file = self.config.get("VECTORSTORE.FAISS.DB_FILE")
        self.collection_name = "index"
    def get_store(self,collection_name=None,embedding=None) -> FAISS:
        if collection_name:
            self.collection_name = collection_name
        if not embedding:
            embedding = self.vectorstoreLib.langchainLib.get_embedding()
        index = faiss.IndexFlatL2(len(embedding.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store
            
    def add_docs(self,vector_store:FAISS,docs,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(docs,batch,vector_store.add_documents)
        return all_ids
    
    def add_texts(self,vector_store:FAISS,texts,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(texts,batch,vector_store.add_texts)
        return all_ids
    
    def delete(self,vectorstore: FAISS,ids: List[str] | None = None):
        return vectorstore.delete(ids) 
      
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
            embedding = self.vectorstoreLib.langchainLib.get_embedding()
        vectorestore = FAISS.load_local(db_file, embeddings=embedding, index_name=collection_name, allow_dangerous_deserialization=True)
        return vectorestore
    
    def search(self,query,vectorstore: FAISS,k=10,filter={}):
        return vectorstore.similarity_search(query,k=k,filter=filter)
    
    def search_with_score(self,query,vectorstore: FAISS,k=10,filter={}):
        return vectorstore.similarity_search_with_score(query,k=k,filter=filter)
