from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

#from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
class PdfLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
    def loader(self, filename: str):
       return PyPDFLoader(filename)
    def load_and_split(self, pdf_file, chunk_size=1000,chunk_overlap=0):
        loader = self.loader(pdf_file)
        docs = loader.load()
        spliter:RecursiveCharacterTextSplitter = self.langchainLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        splited_docs = spliter.split_documents(docs)
        return splited_docs
    