from ylz_utils.langchain.documents import DocumentLib
from langchain_community.document_loaders.image import UnstructuredImageLoader

class ImageLib(DocumentLib):
    def loader(self,file_name):
        return UnstructuredImageLoader(file_name)