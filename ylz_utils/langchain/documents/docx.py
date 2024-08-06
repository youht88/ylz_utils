from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
class DocxLib():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
    def loader(self, filename: str):
        #"./example_data/fake.docx"
       return Docx2txtLoader(filename)
    def load_and_split(self, docx_file, chunk_size=1000,chunk_overlap=0):
        loader = self.loader(docx_file)
        docs = loader.load()
        spliter:RecursiveCharacterTextSplitter = self.langchainLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        splited_docs = spliter.split_documents(docs)
        return splited_docs
    