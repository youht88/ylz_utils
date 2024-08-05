#from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

class PptxLoader():
    ppt = None
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
    def loader(self, filename: str):
        #"./example_data/fake-power-point.pptx"
        self.ppt = Presentation(filename)
        return self
    def lazy_load(self):
        if not self.ppt:
            raise Exception("没有正确指定ppt loader!")
        for slide in self.ppt.slides:
            text = ""
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text_frame = shape.text_frame
                    text = text + text_frame.text + "\n"
            yield Document(text)
    def load(self):
        docs = []
        for doc in self.lazy_load():
            docs.append(doc)
        return docs

    def load_and_split(self, docx_file, chunk_size=1000,chunk_overlap=0):
        loader = self.loader(docx_file)
        docs = loader.load()
        spliter:RecursiveCharacterTextSplitter = self.langchainLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        splited_docs = spliter.split_documents(docs)
        return splited_docs