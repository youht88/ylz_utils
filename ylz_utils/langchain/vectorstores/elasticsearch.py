from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain.vectorstores import VectorstoreLib

from langchain_elasticsearch import ElasticsearchStore,DenseVectorStrategy
from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_core.documents import Document
from tqdm import tqdm
from typing import List


from ylz_utils.config import Config

class ESLib():
    def __init__(self,vectorstoreLib:VectorstoreLib):
        self.config = Config()
        self.vectorstoreLib = vectorstoreLib
        self.client = None
        self.es_host=self.config.get("VECTORSTORE.ES.HOST")
        self.es_user=self.config.get("VECTORSTORE.ES.USER")
        self.es_password=self.config.get("VECTORSTORE.ES.PASSWORD")
    def init_client(self,host=None,es_user=None,es_password=None):
        es_connection = create_elasticsearch_client(
            url=host or self.es_host,
            username=es_user or self.es_user,
            password=es_password or self.es_password,
            params = {"verify_certs":False,"ssl_show_warn":False},
        )
        self.client = es_connection
        return self.client
    
    def get_store(self,index_name="langchain_index",embedding=None) -> ElasticsearchStore:        
        if not embedding:
            embedding = self.vectorstoreLib.langchainLib.get_embedding()
        if not self.client:
            self.init_client()
        return ElasticsearchStore(
                    es_connection=self.client,
                    index_name=index_name,
                    embedding=embedding
                )
    def delete_store(self,index_name):
        # "test-metadata, test-elser, test-basic"
        return self.client.indices.delete(
            index=index_name,
            ignore_unavailable=True,
            allow_no_indices=True
        )
            
    def add_docs(self,vector_store: ElasticsearchStore,docs,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(docs,batch,vector_store.add_documents)
        return all_ids

    def add_texts(self,vector_store: ElasticsearchStore,texts,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(texts,batch,vector_store.add_texts)
        return all_ids
        
    def delete(self,vectorstore: ElasticsearchStore,ids: List[str] | None = None):
        return vectorstore.delete(ids) 
          
    def search(self,query,vectorstore: ElasticsearchStore,k=10,filter=[]):
        return vectorstore.similarity_search(query,k=k,filter=filter)
    
    def search_with_score(self,query,vectorstore: ElasticsearchStore,k=10,filter=[]):
        return vectorstore.similarity_search_with_score(query,k=k,filter=filter)
