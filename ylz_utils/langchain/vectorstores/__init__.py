from abc import abstractmethod
from typing import Literal
from tqdm import tqdm

from langchain_core.documents import Document

from ylz_utils.langchain import LangchainLib

class VectorstoreLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    
    @abstractmethod                
    def get_store(self,collection_name=None,embedding=None):
        pass

    def add_docs(self,*,vectorstore, 
                 docs: list[Document], 
                 batch:int=1,
                 source_hash_key="source_sha256",
                 metadata_filter=None,
                 method:Literal['replace','skip','add']='skip') -> list[str]:
        # 过滤掉None类型和不合法的的metadata
        docs = self.valid_docs_metadata(docs)
        # 如果drop_key不为空，在method=='replace'的情况下删除docs中source_hash256相同的文档。metadata_filter为过滤条件，只有满足metadata_filter的文档才会被删除。
        if source_hash_key:
            finded_ids = self.find_source_sha256(docs=docs,
                                            vectorestore=vectorstore,
                                            source_hash_key=source_hash_key,
                                            metadata_filter=metadata_filter)
            if finded_ids:
                if method =='skip':
                    return finded_ids
                if method == 'replace':
                    self.delete(vectorstore,ids=finded_ids)
        all_ids = self.__split_batch_and_add(docs,batch,vectorstore.add_documents)
        return all_ids

    def add_texts(self,vectorstore, texts,batch:int=1) -> list[str]:
        all_ids = self.__split_batch_and_add(texts,batch,vectorstore.add_texts)
        return all_ids

    def delete(self,vectorstore,ids: list[str] | None = None):
        return vectorstore.delete(ids) 

    def search(self,query,vectorstore, k=10,metadata_filter=None):
        if metadata_filter:
            return vectorstore.similarity_search(query,k=k,filter=metadata_filter)
        else:
            return vectorstore.similarity_search(query,k=k)
    
    def search_with_score(self,query,vectorstore, k=10,metadata_filter=None):
        if metadata_filter:
            return vectorstore.similarity_search_with_relevance_scores(query,k=k,filter=metadata_filter)
        else:
            return vectorstore.similarity_search_with_relevance_scores(query,k=k)

    @abstractmethod
    def find_source_sha256(self,docs:list[Document],vectorestore,source_hash_key:str="source_sha256",metadata_filter={})->bool:
        raise Exception("find_source_sha256 not implemented")

    def valid_docs_metadata(self,docs:list[Document]) -> list[Document]:
        for doc in docs:
            doc.metadata = { k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
        return docs  

    def __split_batch_and_add(self,docs,batch,func) ->  list[list]:
        # ["a","b","c","d","e"] -> [["a","b"],["c","d"],["e"]] when batch is 3
        batch_items = []
        n = len(docs)
        # 特殊情况处理
        if batch <= 0:
            pass  # 如果batch为0或负数，返回空数组
        elif batch == 1:
            batch_items=[docs]  # 如果batch为1，返回原数组的单元素数组
        elif batch >= n:
            batch_items =  [[docs[i]] for i in range(n)]  # 如果batch大于数组长度，返回每个元素的单独数组
        else:
            # 计算每个batch应该包含的基本数量和多余的元素
            base_size = n // batch
            extra = n % batch

            start = 0

            for i in range(batch):
                # 每个batch的大小，如果有多余的元素，则当前batch大小加1
                current_size = base_size + (1 if i < extra else 0)
                end = start + current_size
                batch_items.append(docs[start:end])
                start = end
        all_ids = []
        with tqdm(total= len(batch_items),desc="文档片段",colour="#9999ff") as pbar:
            for index,item_docs in enumerate(batch_items):
                ids = func(item_docs)
                all_ids.extend(ids)
                pbar.update(1)
        return all_ids
