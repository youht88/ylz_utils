
from ylz_utils.langchain.vectorstores.elasticsearch import ESLib
from ylz_utils.langchain.vectorstores.faiss import FaissLib


class VectorstoreLib():
    def __init__(self,langchainLib):
        self.faissLib = FaissLib(langchainLib)
        self.esLib = ESLib(langchainLib)