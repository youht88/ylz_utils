#from langchain_community.document_loaders import UnstructuredPowerPointLoader
from pptx import Presentation

class PptxLoader():
    def loader(self, filename: str):
        #"./example_data/fake-power-point.pptx"
        ppt = Presentation(filename)
        for slide in ppt.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text_frame = shape.text_frame
                    print(text_frame.text)
