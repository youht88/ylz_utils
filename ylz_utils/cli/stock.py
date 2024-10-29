from fastapi import FastAPI
import uvicorn
from ylz_utils.stock.snowball import SnowballStock
from ylz_utils.stock.mairui import MairuiStock


def stock(args):
    host = args.host or '0.0.0.0'
    port = args.port or 8000
    snowballLib:SnowballStock = SnowballStock()
    mairuiLib:MairuiStock = MairuiStock()

    # 创建 FastAPI 实例
    app = FastAPI()
    snowballLib.setup_router()
    mairuiLib.setup_router()
    app.include_router(snowballLib.router)
    app.include_router(mairuiLib.router)

    # 创建一个路由，定义路径和处理函数
    uvicorn.run(app, host=host, port=port)