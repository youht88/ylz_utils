from __future__ import annotations
from typing import TYPE_CHECKING

from ylz_utils.langchain.vectorstores import VectorstoreLib

from langchain_elasticsearch import ElasticsearchStore,DenseVectorStrategy
from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_core.documents import Document
from tqdm import tqdm
from typing import List


from ylz_utils.config import Config

class ESLib(VectorstoreLib):
    def __init__(self,langchainLib,host=None,user=None,password=None):
        super().__init__(langchainLib)
        self.es_host= host or self.config.get("VECTORSTORE.ES.HOST")
        self.es_user= user or self.config.get("VECTORSTORE.ES.USER")
        self.es_password= password or self.config.get("VECTORSTORE.ES.PASSWORD")
        self.client = self.init_client()
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
            embedding = self.langchainLib.get_embedding()
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

    def find_source_sha256(self, docs: list[Document], vectorestore:ElasticsearchStore, source_hash_key: str = "source_sha256", metadata_filter=None)->list[str]:
        if source_hash_key:
            source_sha256_list = list(set([doc.metadata.get(source_hash_key) for doc in docs]))
            if source_sha256_list:
                if metadata_filter:
                    metadata_filter = {"$and":[metadata_filter,{"terms":{f'metadata.{source_hash_key}.keyword':source_sha256_list}}]}
                else:
                    metadata_filter = {"terms":{f'metadata.{source_hash_key}.keyword':source_sha256_list}}
                index_name = vectorestore._store.index
                text_field = vectorestore._store.text_field
                client = vectorestore.client
                try:
                    res = client.search(
                        index=index_name,
                        query={
                            "bool": {
                                "filter": metadata_filter
                            }
                        },
                        size=10000,
                        _source=[text_field]
                    )
                    res_ids = [hit['_id'] for hit in res['hits']['hits']]
                    return res_ids
                except Exception as e:
                    print(f"Error: {e}")
                    return []
        return []
    

