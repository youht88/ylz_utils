from ylz_utils.langchain.vectorstores import VectorstoreLib

from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.documents import Document

class PostgresLib(VectorstoreLib):
    def __init__(self,langchainLib,host=None,port=None,user=None,password=None,database=None):
        super().__init__(langchainLib)
        # See docker command above to launch a postgres instance with pgvector enabled.
        self.host= host or self.config.get("VECTORSTORE.PG.HOST")
        self.port= port or self.config.get("VECTORSTORE.PG.PORT")
        self.user= user or self.config.get("VECTORSTORE.PG.USER")
        self.password= password or self.config.get("VECTORSTORE.PG.PASSWORD")
        self.database= database or self.config.get("VECTORSTORE.PG.DATABASE")
        self.connection = f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"  # Uses psycopg3!
        self.client = self.init_client()   
    
    def init_client(self):
        self.client = PGEngine.from_connection_string(url=self.connection)
        return self.client

    def get_store(self,index_name="langchain_index",embedding=None) -> PGVectorStore:        
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        if not self.client:
            self.init_client()
        return PGVectorStore.create_sync(
            engine=self.client,
            table_name=index_name,
            embedding_service=embedding,
        )

    def delete_store(self,index_name):
        # "test-metadata, test-elser, test-basic"
        return PGVectorStore.drop_vector_index(index_name)

    def find_source_sha256(self, docs: list[Document], vectorestore:PGVectorStore, source_hash_key: str = "source_sha256", metadata_filter=None)->list[str]:
        if source_hash_key:
            source_sha256_list = list(set([doc.metadata.get(source_hash_key) for doc in docs]))
            if source_sha256_list:
                if metadata_filter:
                    metadata_filter = {"$and":[metadata_filter,{source_hash_key:{"$in":source_sha256_list}}]}
                else:
                    metadata_filter = {source_hash_key: {"$in":source_sha256_list}}
                try:
                    res_ids = vectorestore.similarity_search("",k=10000,filter=metadata_filter)['ids']
                    return res_ids
                except Exception as e:
                    print(f"Error: {e}")
                    return []
        return []
    

