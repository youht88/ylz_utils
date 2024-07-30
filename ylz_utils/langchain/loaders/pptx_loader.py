#from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document
from pptx import Presentation

class PptxLoader():
    text = ""
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
        return Document(self.text)