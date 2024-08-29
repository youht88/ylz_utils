from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.life_graph import LifeGraph
class Node():
    def __init__(self,testGraph:LifeGraph):
        self.testGraph = testGraph
    def get_llm(self,llm_key=None,llm_model=None):
        llm_key = llm_key or self.testGraph.llm_key
        llm_model = llm_model or self.testGraph.llm_model
        print("LLM to Used:",llm_key,llm_model)
        return self.testGraph.graphLib.langchainLib.get_llm(llm_key,llm_model)