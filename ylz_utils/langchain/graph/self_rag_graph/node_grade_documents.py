from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.self_rag_graph import SelfRagGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List

class GradeDocumentsNode():
    def __init__(self,self_rag_graph:SelfRagGraph):
        self.self_rag_graph = self_rag_graph
    def get_retrieval_grader(self):
        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | self.self_rag_graph.normal_llm_grader
        #question = "agent memory"
        #docs = self.self_rag_graph.retriever.get_relevant_documents(question)
        #doc_txt = docs[1].page_content
        #print(retrieval_grader.invoke({"question": question, "document": doc_txt}))        
        return retrieval_grader
    
    def __call__(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents:List[Document] = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.get_retrieval_grader().invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}