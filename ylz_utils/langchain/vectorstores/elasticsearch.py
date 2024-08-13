from langchain_elasticsearch import ElasticsearchStore,DenseVectorStrategy
from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_core.documents import Document
from tqdm import tqdm
from typing import List

from ylz_utils.config import Config

class ESLib():
    def __init__(self,langchainLib):
        self.config = Config()
        self.langchainLib = langchainLib
        self.client = None
        self.es_host=self.config.get("ES.HOST")
        self.es_user=self.config.get("ES.USER")
        self.es_password=self.config.get("ES.PASSWORD")
    def init_client(self,host,es_user,es_password):
        es_connection = create_elasticsearch_client(
                url=host or self.es_host,
                username=es_user or self.es_user,
                password=es_password or self.es_password,
                params = {"verify_certs":False,"ssl_show_warn":False},
            )
        self.client = es_connection
    def get_store(self,index_name="langchain_index",embedding=None) -> ElasticsearchStore:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
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
            
    def create_from_docs(self,vectorstore:ElasticsearchStore,docs) -> List[str]:
        all_ids = []
        with tqdm(total= len(docs)) as pbar:
            for idx,doc in enumerate(docs):
                if doc.page_content:
                    ids = vectorstore.add_documents([doc])
                    all_ids.extend(ids)
                pbar.update(1)
        return all_ids
    def create_from_texts(self,vectorstore:ElasticsearchStore,texts) -> List[str]:
        all_ids = []
        with tqdm(total= len(texts)) as pbar:
            for idx,text in enumerate(texts):
                if text:
                    ids = vectorstore.add_texts([text])
                    all_ids.extend(ids)
                pbar.update(1)
        return all_ids
        
    def delete(self,vectorstore: ElasticsearchStore,ids: List[str] | None = None):
        return vectorstore.delete(ids) 
          
    def search(self,query,vectorstore: ElasticsearchStore,k=10,filter=[]):
        return vectorstore.similarity_search(query,k=k,filter=filter)
    
    def search_with_score(self,query,vectorstore: ElasticsearchStore,k=10,filter=[]):
        return vectorstore.similarity_search_with_score(query,k=k,filter=filter)
