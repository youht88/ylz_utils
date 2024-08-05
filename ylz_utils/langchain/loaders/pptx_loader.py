#from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pptx 
from pptx.slide import Slide,SlideLayout,SlideShapes
from pptx.presentation import Presentation
from typing import List,Dict
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches,Cm,Pt

class PptxLoader():
    ppt: Presentation  = None
    slides: List[Slide] = []
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
    def newer(self,filename:str):
        self.ppt = pptx.Presentation()
        self.filename = filename
        return self
    def loader(self, filename: str):
        #"./example_data/fake-power-point.pptx"
        self.ppt = pptx.Presentation(filename)
        self.filename = filename
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
    def add_slide(self,slide_layout=None):
        if not slide_layout:
            slide_layout = self.ppt.slide_layouts[0]
        slide = self.ppt.slides.add_slide(slide_layout)
        self.slides.append(slide)
        return self
    def set_title(self,title,subtitle=None,slide_idx=None):
        if slide_idx:
            slide = self.slides[slide_idx]
        else:
            slide = self.slides[-1]
        shapes = slide.shapes
        shapes.title.text = title
        return self
    def add_text(self,text,left,top,width,height,slide_idx=None):
        if slide_idx:
            slide = self.slides[slide_idx]
        else:
            slide = self.slides[-1]
        shapes = slide.shapes
        txBox = shapes.add_textbox(left,top,width,height)
        tf = txBox.text_frame
        tf.text = text
        return tf
    def add_text_paragraph(self,tf,text,bold=False,font_size=40,level=0):
        p=tf.add_paragraph()
        p.text = text
        p.font.blod = bold
        p.font.size = Pt(font_size)
        p.level = level
        return p
    def add_image(self,slide,image_path,left,top,width,height):
        shapes = slide.shapes
        image = shapes.add_picture(image_path,left,top,width,height)
        return image
    def add_shape(self,slide,shape_type,left,top,width,height,text=None):
        shapes = slide.shapes
        shape = shapes.add_shape(shape_type,left,top,width,height)
        shape.text = text
        return shape
    def add_table(self,slide,contents:List[Dict],left,top,width,height):
        shapes = slide.shapes
        if len(contents)==0:
            raise Exception("表格应至少包含一行一列!")
        rows = len(contents)
        cols = len(contents[0].keys())
        table = shapes.add_table(rows,cols,left,top,width,height).table
        for idx,row in enumerate(contents):
            for jdx,col_key in enumerate(row):
                table.cell(idx,jdx).text = contents[idx][col_key]
        return table
    def save(self):
        self.ppt.save(self.filename)
