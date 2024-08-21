from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_core.documents import Document

class UrlLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
    def loader(self, url, max_depth=2, extractor=None, metadata_extractor=None):
        #"./example_data/fake.docx"
        loader = RecursiveUrlLoader(
            url = url,
            max_depth= max_depth,
            # use_async=False,
            extractor= extractor,
            metadata_extractor= metadata_extractor
            # exclude_dirs=(),
            # timeout=10,
            # check_response_status=True,
            # continue_on_failure=True,
            # prevent_outside=True,
            # base_url=None,
        )
        return loader

    def load_and_split_markdown(self, 
                                url,
                                max_depth=2,
                                extractor=None, 
                                metadata_extractor=None, 
                                chunk_size=1000,
                                chunk_overlap=0) -> list[dict[str,any]]:
        loader = self.loader(url,max_depth=max_depth,extractor=extractor,metadata_extractor=metadata_extractor)
        docs = loader.load()
        transformer = MarkdownifyTransformer()
        converted_docs = transformer.transform_documents(docs)
        result = []
        for doc in converted_docs:
            splited_docs = self.langchainLib.splitterLib.split_markdown_docs(doc.page_content,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            result.append({"doc":doc,"blocks":splited_docs,"metadata":doc.metadata})
        return result
    
    def load_and_split(self,url,max_depth=2, 
                       extractor=None, 
                       metadata_extractor=None, 
                       chunk_size=1000,
                       chunk_overlap=0) -> list[Document]:
        res = self.load_and_split_markdown(url,max_depth=max_depth,
                                           extractor=extractor,
                                           metadata_extractor=metadata_extractor,
                                           chunk_size = chunk_size,
                                           chunk_overlap = chunk_overlap)
        docs = []
        for doc in res:
            for block in doc["blocks"]:
                docs.append(Document(block.page_content,metadata=doc["metadata"]))
        return docs