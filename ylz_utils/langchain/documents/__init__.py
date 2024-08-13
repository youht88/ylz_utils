from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from ylz_utils.langchain.documents.dir_loader import DirLoader
from ylz_utils.langchain.documents.docx import DocxLib
from ylz_utils.langchain.documents.image import ImageLib
from ylz_utils.langchain.documents.pdf import PdfLib
from ylz_utils.langchain.documents.pptx import PptxLib
from ylz_utils.langchain.documents.url import UrlLib


class DocumentLib():
   def __init__(self,langchainLib:LangchainLib):
      self.dir = DirLoader(langchainLib)
      self.url = UrlLib(langchainLib)
      self.docx = DocxLib(langchainLib)
      self.pptx = PptxLib(langchainLib)
      self.image = ImageLib(langchainLib)
      self.pdf = PdfLib(langchainLib)