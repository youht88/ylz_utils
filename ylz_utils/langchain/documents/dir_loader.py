
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langchain_community.document_loaders import DirectoryLoader

class DirLoader():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.loader  = DirectoryLoader