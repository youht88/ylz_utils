from ylz_utils.config import Config
from ylz_utils.langchain import LangchainLib
from ylz_utils.langchain.graph.life_graph import LifeGraph
from ylz_utils.langchain.graph.test_graph import TestGraph

Config.init('ylz_utils')
langchainLib = LangchainLib()

def get_life_graph():
    lifeGraph = LifeGraph(langchainLib)
    lifeGraph.set_nodes_llm_config(("LLM.DEEPBRICKS",None))
    graph = lifeGraph.get_graph()
    return graph

def get_test_graph():
    testGraph = TestGraph(langchainLib)
    testGraph.set_nodes_llm_config(("LLM.DEEPBRICKS",None))
    graph = testGraph.get_graph()
    return graph

