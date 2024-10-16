from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel,Field
from typing import List,Union
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.langchain import LangchainLib
from ylz_utils.langchain.graph.life_graph import LifeGraph
from ylz_utils.langchain.graph.test_graph.function import FunctionGraph,State
from ylz_utils.langchain.graph.test_graph import TestGraph,State as TestState
from ylz_utils.langchain.graph.stock_graph import StockGraph,NewState
from langgraph.graph import MessagesState

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )
    
def serve(args):
    print("args:",args)
    langchainLib: LangchainLib = LangchainLib()
    #langchainLib.add_plugins()    
    path = args.path
    host = args.host
    port = args.port
    llm_key = args.llm_key
    llm_model = args.llm_model
    embedding_key = args.embedding_key
    embedding_model = args.embedding_model
    langchainLib.llmLib.set_default_llm_key(llm_key or "LLM.DEEPBRICKS")
    langchainLib.embeddingLib.set_default_embedding_key(embedding_key or "EMBEDDING.DASHSCOPE")
    llm = langchainLib.get_llm(llm_key)
    
    # neo4jLib = Neo4jLib(None,'neo4j','abcd1234')
    # langchainLib.init_neo4j(neo4jLib)
    # lifeGraph = LifeGraph(langchainLib)
    # lifeGraph.set_nodes_llm_config((llm_key,None))
    # lifeGraph.set_thread("youht","default")
    # life_graph = lifeGraph.get_graph()
    
    testGraph = TestGraph(langchainLib)
    # #testGraph.set_nodes_llm_config({'default':{'llm_key':llm_key,'llm_model':llm_model}})
    # #testGraph.set_thread("youht","default")
    test_graph = testGraph.get_graph()

    stockGraph = StockGraph(langchainLib)
    stock_graph = stockGraph.get_graph()

    app = FastAPI(title="Langserve")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    # chain = langchainLib.get_prompt(human_keys={"input":"问题"}) | langchainLib.get_llm(llm_key,llm_model) | langchainLib.get_outputParser()
    # add_routes(app,runnable=chain,path=path)

    #add_routes(app,runnable=life_graph.with_types(input_type=MessagesState),path="/life")
    #add_routes(app,runnable=test_graph.with_types(input_type=State),path="/test_graph",include_callback_events=True)
    #add_routes(app,runnable=test_graph,path="/test_graph")
    add_routes(app,runnable=stock_graph.with_types(input_type=NewState),path="/stock_graph")
    add_routes(app,runnable=test_graph.with_types(input_type=TestState),path="/test_graph")

    uvicorn.run(app, host = host, port = port)
