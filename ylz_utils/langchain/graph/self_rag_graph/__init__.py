
from ylz_utils.langchain.graph import GraphLib

from langgraph.graph import StateGraph, START,END, MessagesState
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from .state import GraphState,GradeDocuments,GradeHallucinations,GradeAnswer

from .node_grade_documents import GradeDocumentsNode
from .node_retrieve import RetrieveNode
from .node_generate import GenerateNode
from .node_transform_query import TransformQueryNode
from .edge_decide_to_generate import DecideToGenerateEdge
from .edge_grade_generation_v_documents_and_question import GradeGenerationVDocumentsAndQuestionEdge


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class SelfRagGraph(GraphLib):
    retriever = None
    normal_llm_grader = None
    hallucinations_llm_grader = None
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
    
    def set_retriever(self,retriever=None):
        if retriever:
            self.retriever = retriever
        else:
            dbname = "langgraph1.faiss"
            try:
                vectorstore = self.langchainLib.vectorstoreLib.faissLib.load(dbname)
            except:
                url = "https://langchain-ai.github.io/langgraph/how-tos/"
                docs = self.langchainLib.documentLib.url.load_and_split(url=url,max_depth=1,chunk_size=256,chunk_overlap=0)
                print(f"there is {len(docs)} docs to be add to {dbname}.")
                # Add to vectorDB
                print(docs)
                vectorstore = self.langchainLib.vectorstoreLib.faissLib.get_store()
                _ = self.langchainLib.vectorstoreLib.faissLib.add_docs(vectorstore,docs,batch=10)
                vectorstore.save_local(dbname)
            self.retriever = vectorstore.as_retriever()

    def set_llm(self,llm_key,llm_model):
        self.llm = self.langchainLib.get_llm(llm_key,llm_model)
        self.normal_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.hallucinations_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        self.answer_llm_grader = self.llm.with_structured_output(GradeAnswer)

    def _check(self):
        if not self.retriever:
            raise Exception("先调用set_retriever(retriever)")
        if not self.llm or not self.normal_llm_grader or \
           not self.hallucinations_llm_grader or \
           not self.answer_llm_grader:
            raise Exception("先调用self.set_llm(llm_key,llm_model)")

    def get_graph(self):
        self._check()
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", RetrieveNode(self))  # retrieve
        workflow.add_node("grade_documents", GradeDocumentsNode(self))  # grade documents
        workflow.add_node("generate", GenerateNode(self))  # generatae
        workflow.add_node("transform_query", TransformQueryNode(self))  # transform_query

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            DecideToGenerateEdge(self),
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            GradeGenerationVDocumentsAndQuestionEdge(self),
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        graph = workflow.compile(self.memory)
        
        return graph