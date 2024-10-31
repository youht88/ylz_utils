from fastapi.responses import JSONResponse
from ylz_utils.config import Config 

from ylz_utils.loger import LoggerLib
from fastapi import FastAPI, HTTPException
import fastapi_cdn_host

from ylz_utils.stock.snowball import SnowballLib
from ylz_utils.stock.mairui import MairuiLib


if __name__ == "__main__":
    import uvicorn

    Config.init('ylz_utils')
    logger = LoggerLib.init('ylz_utils')
    
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
        
    