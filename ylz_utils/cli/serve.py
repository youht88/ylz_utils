from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from ylz_utils.langchain import LangchainLib

def serve(args):
    print("args:",args)
    langchainLib: LangchainLib = LangchainLib()
    langchainLib.add_plugins()    
    path = args.path
    host = args.host
    port = args.port
    llm_key = args.llm
    model = args.model
    app = FastAPI(title="Langserve")
    chain = langchainLib.get_prompt(human_keys={"input":"问题"}) | langchainLib.get_llm(llm_key,model) | langchainLib.get_outputParser()
    add_routes(app,runnable=chain,path=path)
    uvicorn.run(app, host = host, port = port)
