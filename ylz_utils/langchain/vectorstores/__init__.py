from ylz_utils.langchain.vectorstores.faiss import FaissLib


class VectorstoreLib():
    def __init__(self,langchainLib):
        self.faiss = FaissLib(langchainLib)