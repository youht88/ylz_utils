from ylz_utils.config import Config
from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.langchain import LangchainLib
from ylz_utils.langchain.graph.life_graph import LifeGraph
from ylz_utils.langchain.graph.test_graph import TestGraph

from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import json

Config.init('ylz_utils')
langchainLib = LangchainLib()

with open("mcp_server.json","r") as f:
    mcp_server_config = json.load(f)

def get_life_graph():
    print("graph_cloud:life")
    neo4jLib = Neo4jLib(password="abcd1234")
    langchainLib.init_neo4j(neo4jLib)
    lifeGraph = LifeGraph(langchainLib)
    lifeGraph.set_thread("youht","default")
    lifeGraph.set_nodes_llm_config(("LLM.ZHIPU",None))
    graph = lifeGraph.get_graph()
    return graph

def get_test_graph():
    print("graph_cloud:test")
    testGraph = TestGraph(langchainLib)
    testGraph.set_nodes_llm_config(("LLM.DEEPBRICKS",None))
    graph = testGraph.get_graph()
    return graph

@asynccontextmanager
async def gemini_25_graph():
    async with MultiServerMCPClient(mcp_server_config) as client:
        llm = langchainLib.get_llm(model="google/gemini-2.5-pro-exp-03-25:free")
        agent = create_react_agent(llm,client.get_tools())
        yield agent

@asynccontextmanager
async def deepseek_v3_graph():
    async with MultiServerMCPClient(mcp_server_config) as client:
        llm = langchainLib.get_llm(model="deepseek/deepseek-chat-v3-0324:free")
        agent = create_react_agent(llm,client.get_tools())
        yield agent

#life_graph = get_life_graph()
#test_graph = get_test_graph()
