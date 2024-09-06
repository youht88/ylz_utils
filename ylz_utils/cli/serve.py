from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.langchain import LangchainLib
from ylz_utils.langchain.graph.life_graph import LifeGraph

def serve(args):
    print("args:",args)
    langchainLib: LangchainLib = LangchainLib()
    langchainLib.add_plugins()    
    path = args.path
    host = args.host
    port = args.port
    llm_key = args.llm_key
    llm_model = args.llm_model

    llm = langchainLib.get_llm(llm_key)
    
    neo4jLib = Neo4jLib(None,'neo4j','abcd1234')
    langchainLib.init_neo4j(neo4jLib)
    lifeGraph = LifeGraph(langchainLib)
    lifeGraph.set_nodes_llm_config((llm_key,None))
    lifeGraph.set_thread("youht","default")
    life_graph = lifeGraph.get_graph()
    
    app = FastAPI(title="Langserve")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    chain = langchainLib.get_prompt(human_keys={"input":"问题"}) | langchainLib.get_llm(llm_key,llm_model) | langchainLib.get_outputParser()
    add_routes(app,runnable=chain,path=path)

    add_routes(app,runnable=life_graph,path="/life")

    uvicorn.run(app, host = host, port = port)
