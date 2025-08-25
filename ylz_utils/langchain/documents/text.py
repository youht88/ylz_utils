from langchain_community.document_loaders import TextLoader

from ylz_utils.langchain.documents import DocumentLib

class TextLib(DocumentLib):
    def loader(self, file_name: str):
        return TextLoader(file_name,autodetect_encoding=True)