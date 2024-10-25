from datetime import datetime
import logging
import re
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import requests
from ylz_utils.config import Config 

from rich import print
from ylz_utils.database.elasticsearch import ESLib

import concurrent.futures
import time

from ylz_utils.loger import LoggerLib

class StockLib:
    stock:list = []
    def __init__(self):
        self.esLib = ESLib(using='es')
        # 获取当前模块的目录
        self.gpdm = None
        self.zsdm = None
        self.scheduler = BackgroundScheduler()
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
    import time
    from ylz_utils.stock.snowball import SnowballStock
    from ylz_utils.stock.mairui.mairui_hszg import HSZG
    from ylz_utils.stock.mairui.mairui_zs import ZS
    from ylz_utils.stock.mairui.mairui_hizj import HIZJ

    Config.init('ylz_utils')
    logger = LoggerLib.init('ylz_utils')
    print("雪球--->")

    lib:SnowballStock = SnowballStock()
    
    # print(lib.watch_list())
    # print(lib.watch_stock(-10))

    # for i in range(10):
    #     res1 = lib._parallel_execute(lib.pankou,['全志科技','欧菲光'])
    #     print(f"****** {i} , {len(res1)}  ******")
    #     print(res1[0]['t'],res1[0]['mr_code'],res1[0]['vc'],res1[0]['vb'])
    #     #lib.esLib.save("capital_flow",[res1],ids=['mr_code','t'])
    #     time.sleep(3)
    
    #lib.esLib.drop("snowball_ssjy_20241025")
    res = lib.quotec_detail("上证指数")
    codes=['全志科技','瑞芯微','欧菲光',"永辉超市","乐鑫科技","联创电子","万达信息","银邦股份","蒙草生态","拉卡拉",
           "新华传媒","宗申动力","隆基绿能","常山北明","旗天科技","国泰君安",
           "国新健康","普洛药业","隆基绿能","中船应急","福日电子","立讯精密","华力创通","中粮资本",
           "东方财富","中国中免","国金证券",
           "中际旭创","小商品城","源杰科技",
           "新元科技","金三江","海达股份","科创新源","华盛锂电","矩阵股份","民德电子","帝尔激光","宇邦新材","乾照光电",
           "保变电气","新诺威","珠江啤酒","国电电力","协创数据","神宇股份","北新建材","未名医药","蜂助手","如通股份",
           "锐捷网络","吉比特","宁德时代","迈为股份","中熔电气","同花顺","光智科技","韦尔股份",
           "上证指数","深证成指","创业板指","中证500"
           ]
    #print([lib._get_stock_code(code) for code in codes])
    # print(HSZG().get_hszg_gg('gn_ldc'))
    # print(HSZG().get_hszg_zg('宗申动力'))
    kwargs = {
        "func":lib.quotec_detail,
        "codes":codes,
        "sync_es":True
    }
    lib.scheduler.add_job(lib.parallel_execute, trigger=CronTrigger(hour='9',minute='30-59',second='*/3'),kwargs=kwargs)
    lib.scheduler.add_job(lib.parallel_execute, trigger=CronTrigger(hour='10',minute='00-59',second='*/3'),kwargs=kwargs)
    lib.scheduler.add_job(lib.parallel_execute, trigger=CronTrigger(hour='11',minute='00-30',second='*/3'),kwargs=kwargs)
    lib.scheduler.add_job(lib.parallel_execute, trigger=CronTrigger(hour='13-14',minute='00-59',second='*/3'),kwargs=kwargs)
    lib.scheduler.add_job(lib.parallel_execute, trigger=CronTrigger(hour='9',minute='30-59',second='*/3'),kwargs=kwargs)
    lib.scheduler.start()

    hizjLib = HIZJ()
    hizjLib.scheduler.add_job(hizjLib.get_hizj_bk, 
                              trigger=CronTrigger(hour='16',day_of_week='mon-fri'),kwargs={"sync_es":True})
    
    # 循环执行任务
    logging.info("hello world")
    try:
        while True:
            try:
                res = lib.esLib.sql_as_df("select count(*) from snowball_ssjy_20241025")
                print("result=",res)        
            except Exception as e:
                logging.error(str(e))
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        lib.scheduler.shutdown()
        
    