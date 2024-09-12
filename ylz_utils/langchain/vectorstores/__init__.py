
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from tqdm import tqdm

#from ylz_utils.langchain.vectorstores.elasticsearch import ESLib
from ylz_utils.langchain.vectorstores.faiss import FaissLib
from ylz_utils.langchain.vectorstores.chroma import ChromaLib


class VectorstoreLib():
    def __init__(self,langchainLib:LangchainLib,provider:Optional[Literal['es','faiss','chroma']]=None):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
        self.faissLib = FaissLib(self)
        #self.esLib = ESLib(self)
        self.chromaLib = ChromaLib(self)
        if provider:
            self.provider = provider
        else:
            self.provider = self.config.get("VECTORSTORE.DEFAULT")
    
    def get_store_with_provider_and_indexname(self, provider_and_indexname,embedding=None):
        provider_indexname = provider_and_indexname.split(":")
        if len(provider_indexname)==2:
            provider = provider_indexname[0]
            indexname = provider_indexname[1]
        else:
            provider = None
            indexname = provider_indexname[0]
        vector_store = None
        if  provider=='faiss':
            try:
                vector_store = self.faissLib.load(embedding=embedding,collection_name=indexname)
            except:
                vector_store = self.faissLib.get_store(collection_name=indexname,embedding=embedding)
        else:    
            vector_store = self.get_store(provider,indexname,embedding)
        return vector_store
    
    def get_store(self,provider:Optional[Literal['es','faiss','chroma']],collection_name=None,embedding=None):
        if not provider:
            provider = self.provider
        if not provider:
            raise Exception("请设置vectorstore provider")
        
        if provider=='es':
            return self.esLib.get_store(collection_name,embedding)
        elif provider == 'faiss':
            return self.faissLib.get_store(collection_name,embedding)
        elif provider == 'chroma':
            return self.chromaLib.get_store(collection_name,embedding)
        else:
            return None
    def add_docs(self,vector_store,docs,batch=1) -> list[str]:
        return self._split_batch_and_add(docs,batch,vector_store.add_documents)
    def add_texts(self,vector_store,texts,batch=1) -> list[str]:
        return self._split_batch_and_add(texts,batch,vector_store.add_texts)
    
    def search(self,query,vectorstore,k=10,filter={}):
        return vectorstore.similarity_search(query,k=k,filter=filter)
    
    def search_with_score(self,query,vectorstore,k=10,filter={}):
        return vectorstore.similarity_search_with_score(query,k=k,filter=filter)
    
    def _split_batch_and_add(self,docs,batch,func) ->  list[list]:
        # ["a","b","c","d","e"] -> [["a","b"],["c","d"],["e"]] when batch is 3
        batch_items = []
        n = len(docs)
        # 特殊情况处理
        if batch <= 0:
            pass  # 如果batch为0或负数，返回空数组
        elif batch == 1:
            batch_items=docs  # 如果batch为1，返回原数组的单元素数组
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
        print(f"size = {len(docs)},parts = {[len(item) for item in batch_items]}")
        all_ids = []
        with tqdm(total= len(batch_items),desc="文档片段",colour="#9999ff") as pbar:
            for index,items in enumerate(batch_items):
                ids = func(items)
                all_ids.extend(ids)
                pbar.update(1)
        return all_ids
