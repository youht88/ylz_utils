#from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

class PptxLoader():
    text = ""
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
    def loader(self, filename: str):
        #"./example_data/fake-power-point.pptx"
        ppt = Presentation(filename)
        text = ""
        for slide in ppt.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text_frame = shape.text_frame
                    text = text + text_frame.text + "\n"
        self.text = text
        return self
    def load(self):
        return [Document(self.text)]
    def load_and_split(self, docx_file, chunk_size=1000,chunk_overlap=0):
        loader = self.loader(docx_file)
        docs = loader.load()
        spliter:RecursiveCharacterTextSplitter = self.langchainLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        splited_docs = spliter.split_documents(docs)
        return splited_docs