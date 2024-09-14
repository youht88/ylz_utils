from ylz_utils.config import Config
from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.langchain import LangchainLib
from ylz_utils.langchain.graph.life_graph import LifeGraph
from ylz_utils.langchain.graph.test_graph import TestGraph

Config.init('ylz_utils')
langchainLib = LangchainLib()

def get_life_graph():
    print("graph_cloud:life")
    neo4jLib = Neo4jLib(password="abcd1234")
    langchainLib.init_neo4j(neo4jLib)
    lifeGraph = LifeGraph(langchainLib)
    lifeGraph.set_thread("youht","default")
    lifeGraph.set_nodes_llm_config(("LLM.DEEPBRICKS",None))
    graph = lifeGraph.get_graph()
    return graph

def get_test_graph():
    print("graph_cloud:test")
    testGraph = TestGraph(langchainLib)
    testGraph.set_nodes_llm_config(("LLM.DEEPBRICKS",None))
    graph = testGraph.get_graph()
    return graph

life_graph = get_life_graph()
test_graph = get_test_graph()