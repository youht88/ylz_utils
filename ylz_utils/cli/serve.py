from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from ylz_utils.langchain_utils import LangchainLib

async def serve(args):
    #print("args:",args)
    langchainLib: LangchainLib = LangchainLib()
    path = args.path
    host = args.host
    port = args.port
    app = FastAPI(title="Langserve")
    chain = langchainLib.get_prompt(human_keys={"input":"问题"}) | langchainLib.get_llm() | langchainLib.get_outputParser()
    add_routes(app,runnable=chain,path=path)
    await uvicorn.run(app, host = host, port = port)