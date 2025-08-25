from ylz_utils.langchain.documents import DocumentLib
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader

class PdfLib(DocumentLib):
    def loader(self, file_name: str):
        return PyPDFLoader(file_name)