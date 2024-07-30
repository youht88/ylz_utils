from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

class FaissLib():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib

    def create_from_docs(self,docs,embedding=None) -> FAISS:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        #vectorstore = FAISS.from_documents(docs,embedding=embedding)
        print("embedding=",embedding)
        vectorstore = FAISS.from_documents([Document("hello")],embedding) 
        print("vectorstore=",vectorstore)  
        with tqdm(total= len(docs)) as pbar:
            for index,doc in enumerate(docs):
                vectorstore.add_documents([doc],embedding=embedding)
                pbar.update(1)
        return vectorstore
    def create_from_textes(self,textes,embedding=None) -> FAISS:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        vectorstore = FAISS.from_texts(textes, embedding=embedding)
        return vectorstore
    
    def delete(self,vectorstore: FAISS,ids):
        return vectorstore.delete(ids) 
      
    def save(self,  db_file:str, vectorstore: FAISS,index_name:str = "index"):
        vectorstore.save_local(db_file,index_name)

    def load(self, db_file:str ,embedding=None, index_name:str = "index") -> FAISS:
        if not embedding:
            embedding = self.langchainLib.get_embedding()
        vectorestore = FAISS.load_local(db_file, embeddings=embedding, index_name=index_name, allow_dangerous_deserialization=True)
        return vectorestore
    
    def search(self,query,vectorstore: FAISS,k=10):
        return vectorstore.similarity_search(query,k=k)
