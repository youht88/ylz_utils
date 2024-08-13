from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langchain_community.document_loaders.image import UnstructuredImageLoader

class ImageLib():
     def __init__(self,langchainLib:LangchainLib):
          self.langchainLib = langchainLib
     def loader(self,filename):
         "./example_data/layout-parser-paper-screenshot.png"
         return UnstructuredImageLoader(filename)