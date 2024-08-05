from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
from typing import List

class FaissLib():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib

    def create_from_docs(self,docs,embedding=None) -> FAISS:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        #vectorstore = FAISS.from_documents(docs,embedding=embedding)
        vectorstore = FAISS.from_documents([Document(" ")],embedding) 
        #vectorstore.add_documents(docs,embedding=embedding)
        all_ids = []
        with tqdm(total= len(docs)) as pbar:
            for index,doc in enumerate(docs):
                if doc.page_content:
                    ids = vectorstore.add_documents([doc],embedding=embedding)
                    all_ids.extend(ids)
                pbar.update(1)
        return vectorstore,all_ids
    def create_from_textes(self,textes,embedding=None) -> FAISS:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        vectorstore = FAISS.from_texts(textes, embedding=embedding)
        return vectorstore
    
    def delete(self,vectorstore: FAISS,ids: List[str] | None = None):
        return vectorstore.delete(ids) 
      
    def save(self,  db_file:str, vectorstore: FAISS,index_name:str = "index"):
        vectorstore.save_local(db_file,index_name)

    def load(self, db_file:str ,embedding=None, index_name:str = "index") -> FAISS:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        vectorestore = FAISS.load_local(db_file, embeddings=embedding, index_name=index_name, allow_dangerous_deserialization=True)
        return vectorestore
    
    def search(self,query,vectorstore: FAISS,k=10):
        vectorstore.similarity_search_with_score
        return vectorstore.similarity_search(query,k=k)
