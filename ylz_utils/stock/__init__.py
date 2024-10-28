from datetime import datetime
import logging
import re
import requests
from ylz_utils.config import Config 

from rich import print
from ylz_utils.database.elasticsearch import ESLib

import concurrent.futures
import time

from ylz_utils.loger import LoggerLib

from fastapi import FastAPI,Request,Response,APIRouter

class StockLib:
    stock:list = []
    def __init__(self):
        self.esLib = ESLib(using='es')
        # 获取当前模块的目录
        self.gpdm = None
        self.zsdm = None
        self.mairui_token = Config.get('STOCK.MAIRUI.TOKEN')
        print("MAIRUI_TOKEN=",self.mairui_token)
        self.mairui_api_url = "http://api.mairui.club" 
    # 定义一个执行函数的方法k
    def parallel_execute(self,**kwargs):
        func = kwargs.pop("func")
        codes = kwargs.get("codes")
        if func:
            logging.info(f"run {func.__name__} as {datetime.now()}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if codes:
                    kwargs.pop("codes")
                    futures = [executor.submit(func,code,**kwargs) for code in codes]
                else:
                    futures = [executor.submit(func,**kwargs)]
                results = [future.result() for future in futures]
                return results
    def _get_bk_code(self,bk_name:str)->dict:
        if not self.bkdm:
            res = requests.get(f"{self.mairui_api_url}/hszg/list/{self.mairui_token}")
            self.bkdm = res.json()
        code_info = list(filter(lambda item:item['code']==bk_name,self.zsdm))
        if code_info:
            if len(code_info)>1:
                print("code_info",code_info)
                raise Exception('板块代码不唯一，请重新配置!')
            return code_info[0]
        else:
            code_info = list(filter(lambda item:item['name'].find(bk_name)>=0,self.zsdm)) 
            if code_info:
                if len(code_info)>1:
                    print("code_info",code_info)
                    raise Exception('板块代码不唯一，请重新配置!')
                else:
                    return code_info[0]
            else:
                raise Exception('没有找到相关板块!')

    def _get_stock_code(self,stock_name:str)->dict:
        """根据股票或指数名称获取股票/指数代码"""
        if not self.gpdm:
            try:
                res = requests.get(f"{self.mairui_api_url}/hslt/list/{self.mairui_token}")
                stock_dm = res.json()
                #合并沪深两市指数代码
                res = requests.get(f"{self.mairui_api_url}/zs/sh/{self.mairui_token}")
                sh_dm = [{**item , "dm":item['dm'].replace('sh','')} for item in res.json()]
                res = requests.get(f"{self.mairui_api_url}/zs/sz/{self.mairui_token}")
                sz_dm = [{**item , "dm":item['dm'].replace('sz','')} for item in res.json()]
                all_dm = stock_dm + sh_dm + sz_dm
                self.gpdm = [
                    {"ts_code":f"{item['dm']}.{item['jys'].upper()}","symbol":f"{item['dm']}","name":item['mc']} for item in all_dm
                ]
            except Exception as e:
                raise Exception(f"获取{stock_name}代码错误,{e}")
        if stock_name.startswith('sh') or stock_name.startswith('sz'):
            # mairui code
            jys = stock_name[:2]
            code = stock_name[2:]
            ts_code = f"{code}.{jys.upper()}"
            mr_code = stock_name
            ball_code = f"{jys.upper()}{code}"
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                name = stock_info[0]['name']
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        elif stock_name.endswith('.SH') or stock_name.endswith('.SZ'):
            #tu-share code
            jys = stock_name[-2:].lower()
            code = stock_name[:-3]
            ts_code = stock_name
            mr_code = f"{jys}{code}"
            ball_code = f"{jys.upper()}{code}"
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                name = stock_info[0]['name']
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        elif stock_name.startswith('SH') or stock_name.startswith('SZ'):
            #雪球code
            jys = stock_name[:2].lower()
            code = stock_name[2:]
            ball_code = stock_name
            ts_code = f"{code}.{jys.upper()}"
            mr_code = f"{jys}{code}"
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                name = stock_info[0]['name']
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        elif stock_name.isnumeric():
            code = stock_name
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                ts_code = stock_info[0]['ts_code']
                jys = ts_code[-2:].lower()
                mr_code = f"{jys}{code}"
                name = stock_info[0]['name']
                ball_code = f"{jys.upper()}{code}"
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        else:
            stock_info = list(filter(lambda item:item["name"]==stock_name.upper(),self.gpdm))
            if stock_info:
                ts_code = stock_info[0]['ts_code']
                jys = ts_code[-2:].lower()
                code = stock_info[0]['symbol']
                name = stock_info[0]['name']
                mr_code = f"{jys}{code}"
                ball_code = f"{jys.upper()}{code}"
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                zs_info = list(filter(lambda item:item["mc"]==stock_name.upper(),self.zsdm))
                if zs_info:
                    mr_code = zs_info[0]['dm']
                    jys = zs_info[0]['jys']
                    name = zs_info[0]['mc']
                    code = mr_code[2:]
                    ts_code = f"{code}.{jys.upper()}"
                    ball_code = f"{jys.upper()}{code}"
                    return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
    

if __name__ == "__main__":
    import uvicorn
    from ylz_utils.stock.snowball import SnowballStock
    from ylz_utils.stock.mairui import MairuiStock

    Config.init('ylz_utils')
    logger = LoggerLib.init('ylz_utils')
    
    snowballLib:SnowballStock = SnowballStock()
    mairuiLib:MairuiStock = MairuiStock()

    # 创建 FastAPI 实例
    app = FastAPI()
    snowballLib.setup_router()
    mairuiLib.setup_router()
    app.include_router(snowballLib.router)
    app.include_router(mairuiLib.router)

    # 创建一个路由，定义路径和处理函数
    uvicorn.run(app, host="127.0.0.1", port=8000)
        
    