import re
from langchain_core.documents import Document

import numpy as np
import openai
from langchain_core.tools import tool

import requests

from ylz_utils.langchain import LangchainLib

response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

#docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]
docs = [Document(txt) for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


#retriever = VectorStoreRetriever.from_docs(docs, openai.Client())
langchainLib = LangchainLib()
embedding = langchainLib.embeddingLib.get_embedding(key="EMBEDDING.OLLAMA")
retriever = langchainLib.vectorstoreLib.faiss.create_from_docs(docs,embedding)


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    #docs = retriever.query(query, k=2)
    docs = langchainLib.vectorstoreLib.faiss.search(query,retriever,k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])