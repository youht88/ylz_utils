from langchain_community.document_loaders import Docx2txtLoader

from ylz_utils.langchain.documents import DocumentLib

class DocxLib(DocumentLib):
    def loader(self, file_name: str):
        return Docx2txtLoader(file_name)