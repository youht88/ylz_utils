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
        self.es_host=self.config.get("ES.HOST")
        self.es_user=self.config.get("ES.USER")
        self.es_password=self.config.get("ES.PASSWORD")
    def init_client(self,host=None,es_user=None,es_password=None,connect_string:str=None):
        if connect_string:
            if connect_string.startswith("es:///"):
                es_url = self.config.get("ES.HOST")
                es_username = self.config.get("ES.USER")
                es_password = self.config.get("ES.PASSWORD")
                index_name = connect_string.split(':///')[1]
            else:
                es_url = f"https://{connect_string.split('@')[1].split('/')[0]}"
                es_username = connect_string.split('@')[0].split('://')[1].split(":")[0]
                es_password = connect_string.split('@')[0].split('://')[1].split(":")[1]
                index_name = connect_string.split('@')[1].split('/')[1]
            es_connection = create_elasticsearch_client(
                url=es_url or self.es_host,
                username=es_username or self.es_user,
                password=es_password or self.es_password,
                params = {"verify_certs":False,"ssl_show_warn":False},
            )
        else:
            es_connection = create_elasticsearch_client(
                url=host or self.es_host,
                username=es_user or self.es_user,
                password=es_password or self.es_password,
                params = {"verify_certs":False,"ssl_show_warn":False},
            )
            index_name = None
        self.client = es_connection
        return self.client,index_name
    
    def get_store(self,index_name="langchain_index",embedding=None) -> ElasticsearchStore:        
        if not embedding:
            embedding = self.vectorstoreLib.langchainLib.get_embedding()
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
            
    def create_from_docs(self,vector_store:ElasticsearchStore,docs,batch:int=1) -> List[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(docs,batch,vector_store.add_documents)
        return all_ids
    def create_from_texts(self,vector_store:ElasticsearchStore,texts,batch:int=1) -> List[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(texts,batch,vector_store.add_texts)
        return all_ids
    def add_docs_to_vectorstore(self,vector_store: ElasticsearchStore,docs,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(docs,batch,vector_store.add_documents)
        return all_ids

    def add_texts_to_vectorstore(self,vector_store: ElasticsearchStore,texts,batch:int=1) -> list[str]:
        all_ids = self.vectorstoreLib._split_batch_and_add(texts,batch,vector_store.add_texts)
        return all_ids
        
    def delete(self,vectorstore: ElasticsearchStore,ids: List[str] | None = None):
        return vectorstore.delete(ids) 
          
    def search(self,query,vectorstore: ElasticsearchStore,k=10,filter=[]):
        return vectorstore.similarity_search(query,k=k,filter=filter)
    
    def search_with_score(self,query,vectorstore: ElasticsearchStore,k=10,filter=[]):
        return vectorstore.similarity_search_with_score(query,k=k,filter=filter)
