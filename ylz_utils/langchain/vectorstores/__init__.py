
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from tqdm import tqdm

from ylz_utils.langchain.vectorstores.elasticsearch import ESLib
from ylz_utils.langchain.vectorstores.faiss import FaissLib


class VectorstoreLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.faissLib = FaissLib(self)
        self.esLib = ESLib(self)
        
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
