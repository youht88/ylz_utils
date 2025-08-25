from ylz_utils.langchain.vectorstores import VectorstoreLib
from langchain_core.documents import Document
import logging

try:
    import chromadb
    from langchain_chroma import Chroma
except:
    logging.warning(f"use `pip install chromadb langchain-chroma` ,otherwise you can't use provider by ChromaLib")


class ChromaLib(VectorstoreLib):
    def __init__(self,langchainLib,host=None,port=None,db_file=None):
        super().__init__(langchainLib)
        self.host = host or self.config.get("VECTORSTORE.CHROMA.HOST")
        self.port = port or self.config.get("VECTORSTORE.CHROMA.PORT")
        self.db_file = db_file or self.config.get("VECTORSTORE.CHROMA.DB_FILE")
        self.use_server = self.config.get("VECTORSTORE.CHROMA.USE_SERVER")
    def get_store(self,collection_name=None,embedding=None) -> Chroma:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        if self.use_server:
            client = chromadb.HttpClient(host=self.host,port=self.port)
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                client=client,  # Where to save data locally, remove if not neccesary
            )
        else:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                persist_directory=self.db_file,  # Where to save data locally, remove if not neccesary
            )
            print("embedding_function=",embedding)
        return vectorstore

    def find_source_sha256(self, docs: list[Document], vectorestore:Chroma, source_hash_key: str = "source_sha256", metadata_filter=None)->list[str]:
        if source_hash_key:
            source_sha256_list = list(set([doc.metadata.get(source_hash_key) for doc in docs]))
            if source_sha256_list:
                if metadata_filter:
                    metadata_filter = {"$and":[metadata_filter,{source_hash_key:{"$in":source_sha256_list}}]}
                else:
                    metadata_filter = {source_hash_key: {"$in":source_sha256_list}}
                res_ids = vectorestore.get(where=metadata_filter)['ids']
                return res_ids
        return []
              
