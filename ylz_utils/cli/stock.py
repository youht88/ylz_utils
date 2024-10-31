from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import fastapi_cdn_host
import uvicorn
from ylz_utils.stock import SnowballLib,MairuiLib


def stock(args):
    host = args.host or '0.0.0.0'
    port = args.port or 8000
    snowballLib:SnowballLib = SnowballLib()
    mairuiLib:MairuiLib = MairuiLib()

    # 创建 FastAPI 实例
    app = FastAPI()
    fastapi_cdn_host.patch_docs(app)
    snowballLib.register_app(app)
    mairuiLib.register_app(app)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(content={"error": exc.detail}, status_code=exc.status_code)
    # 创建一个路由，定义路径和处理函数
    uvicorn.run(app, host=host, port=port)