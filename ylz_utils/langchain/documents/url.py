
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_core.documents import Document

from ylz_utils.langchain.documents import DocumentLib
from ylz_utils.crypto import HashLib

class UrlLib(DocumentLib):
    def loader(self, url, max_depth=2, extractor=None, metadata_extractor=None):
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
    def load(self, url, max_depth=2, extractor=None, metadata_extractor=None,metadata=None):
        loader = self.loader(url, max_depth=max_depth, extractor=extractor, metadata_extractor=metadata_extractor)
        docs = loader.load()
        for doc in  docs:
            file_hash = HashLib.sha256(doc.page_content)
            doc.metadata.update({"source_sha256":file_hash})
            if metadata:
                doc.metadata.update(metadata)  
        return docs

    def load_and_split_markdown(self, 
                                url,
                                metadata=None,
                                max_depth=2,
                                extractor=None, 
                                metadata_extractor=None, 
                                chunk_size=1000,
                                chunk_overlap=0) -> list[dict[str,any]]:
        docs = self.load(url,max_depth,extractor,metadata_extractor,metadata)
        transformer = MarkdownifyTransformer()
        converted_docs = transformer.transform_documents(docs)
        result = []
        for doc in converted_docs:
            splited_docs = self.langchainLib.splitterLib.split_markdown_docs(doc.page_content,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            result.append({"doc":doc,"blocks":splited_docs,"metadata":doc.metadata})
        return result
    
    def load_and_split(self,url,metadata=None,max_depth=2, 
                       extractor=None, 
                       metadata_extractor=None, 
                       chunk_size=1000,
                       chunk_overlap=0) -> list[Document]:
        res = self.load_and_split_markdown(url,metadata=metadata,max_depth=max_depth,
                                           extractor=extractor,
                                           metadata_extractor=metadata_extractor,
                                           chunk_size = chunk_size,
                                           chunk_overlap = chunk_overlap)
        docs = []
        for doc in res:
            for block in doc["blocks"]:
                docs.append(Document(block.page_content,metadata=doc["metadata"]))
        return docs