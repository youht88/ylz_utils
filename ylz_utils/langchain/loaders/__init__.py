from ylz_utils.langchain.loaders.docx_loader import DocxLoader
from ylz_utils.langchain.loaders.image_loader import ImageLoader
from ylz_utils.langchain.loaders.pptx_loader import PptxLoader
from ylz_utils.langchain.loaders.url_loader import UrlLoader


class LoaderLib():
   def __init__(self):
      self.url = UrlLoader()
      self.docx = DocxLoader()
      self.pptx = PptxLoader()
      self.image = ImageLoader()