
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from ylz_utils.langchain.vectorstores.elasticsearch import ESLib
from ylz_utils.langchain.vectorstores.faiss import FaissLib


class VectorstoreLib():
    def __init__(self,langchainLib:LangchainLib):
        self.faissLib = FaissLib(langchainLib)
        self.esLib = ESLib(langchainLib)