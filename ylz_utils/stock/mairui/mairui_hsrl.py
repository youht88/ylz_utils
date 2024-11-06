from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
import requests
from .mairui_base import MairuiBase

class HSRL(MairuiBase):
    def __init__(self):
        super().__init__()
        self.register_router()
    def get_hsrl_ssjy(self,code:str,sync_es:bool=False):
        """获取某个股票的实时交易数据"""
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        date_fields = ['t']
        name = f"hsrl_ssjy"
        keys = ["mr_code","t"]
        df = self.load_data(name,f"hsrl/ssjy/{code}",
                                        add_fields=add_fields,
                                        keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','t'])
            print(f"errors:{es_result["errors"]}")
        return df
    def get_hsrl_mmwp(self,code:str,sync_es:bool=False):
        """获取某个股票的盘口交易数据,返回值没有当前股价，仅有5档买卖需求量价以及委托统计"""
        #数据更新：交易时间段每2分钟
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        date_fields = ['t']
        name = f"hsrl_mmwp"
        keys= ["mr_code","t"]
        df = self.load_data(name,f"hsrl/mmwp/{code}",
                                        add_fields=add_fields,
                                        keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','t'])
            print(f"errors:{es_result["errors"]}")
        return df

    def get_hsrl_zbjy(self,code:str,sync_es:bool=False):
        """获取某个股票的当天逐笔交易数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>21:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"mr_code":mr_code,"ud":ud}
        date_fields = ['t','ud']
        name = f"hsrl_zbjy"
        keys = ["mr_code","t"]
        sql = f"select * from {name} where mr_code='{mr_code}' and strftime('%Y-%m-%d',ud)='{ud}'"
        df = self.load_data(name,f"hsrl/zbjy/{code}",
                                        add_fields=add_fields,
                                        sql=sql,
                                        keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','t'])
            print(f"errors:{es_result["errors"]}")
        return df
    def get_hsrl_fscj(self,code:str):
        """获取某个股票的当天分时成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/fscj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_fjcj(self,code:str):
        """获取某个股票的当天分价成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/fjcj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_zbdd(self,code:str,sync_es:bool=False):
        """获取某个股票的当天逐笔超400手的大单成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>21:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"mr_code":mr_code,"ud":ud}
        date_fields = ['ud']
        skip_condition = f"mr_code == '{mr_code}' & (ud.dt.strftime('%Y-%m-%d')='{ud}')"
        name = f"hsrl_zbdd_{mr_code}"
        df = self._load_data(name,f"hsrl/zbdd/{code}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code","d"],date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','d','t'])
            print(f"errors:{es_result["errors"]}")
        return df
    
    def register_router(self):
        @self.router.get("/hsrl/zbjy/{code}",response_class=HTMLResponse)
        async def get_hsrl_zbjy(code:str,req:Request):
            """获取某个股票的当天逐笔交易数据"""
            try:
                df = self.get_hsrl_zbjy(code)
                df = self._prepare_df(df,req)
                content = self._to_html(df)
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")