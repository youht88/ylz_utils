from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.self_rag_graph import SelfRagGraph

class RetrieveNode():
    def __init__(self,self_rag_graph:SelfRagGraph):
        self.self_rag_graph = self_rag_graph

    def __call__(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        print(state)
        # Retrieval
        documents = self.self_rag_graph.retriever.get_relevant_documents(question)

        return {"documents": documents, "question": question}
