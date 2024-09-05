
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.self_rag_graph import SelfRagGraph

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

class GenerateNode():
    def __init__(self,self_rag_graph:SelfRagGraph):
        self.self_rag_graph = self_rag_graph

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
       
    def __call__(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Prompt
        prompt = hub.pull("rlm/rag-prompt")
        # Chain
        rag_chain = prompt | self.self_rag_graph.llm | StrOutputParser()

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

