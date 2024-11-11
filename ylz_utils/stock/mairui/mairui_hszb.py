from datetime import datetime, timedelta
from typing import Literal
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import requests
from .mairui_base import MairuiBase

class HSZB(MairuiBase):
    def __init__(self):
        super().__init__()
        self.register_router()
    def get_hszb_fsjy(self,code:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m",sync_es:bool=False):
        """获取股票代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        mc = code_info['name']
        add_fields = {"mr_code":mr_code,"fsjb":fsjb,"mc":mc}
        date_fields = ['d']
        keys=["mr_code","fsjb","d"]
        res = requests.get(f"{self.mairui_api_url}/hszb/fsjy/{code}/{fsjb}/{self.mairui_token}")
        data = res.json()
        df = pd.DataFrame([data])
        for key in add_fields:
            df[key] = add_fields[key]
        for item in date_fields:
            df[item] = pd.to_datetime(df[item])
        df_sql = pd.read_sql(f"select * from hszb_fsjy_{fsjb} where mr_code='{mr_code}'",self.sqlite)
        df_sql['d'] = pd.to_datetime(df_sql['d'])
        all_df = pd.concat([df_sql,df],ignore_index=True).drop_duplicates(subset=['mr_code', 'd'])
        def lxxd(window):
            #print(window)
            tag = ((100+window).prod())/100/100/100/100/100
            return tag
        #all_df['tag'] = all_df[['zd','c']].rolling(window=5).apply(lxxd,raw=False)
        all_df['tag'] = all_df['zd'].rolling(window=5).apply(lxxd, raw=False)
        return all_df


    def get_hszb_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易的平均移动线数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszb/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_hszbl_fsjy(self,code:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m",sync_es:bool=False):
        """获取股票代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
    
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        mc = code_info['name']
        add_fields = {"mr_code":mr_code,"fsjb":fsjb,"mc":mc}
        date_fields = ['d']
        keys=["mr_code","fsjb","d"]
        name = f"hszb_fsjy_{fsjb}"
        sql = f"delete from {name} where mr_code='{mr_code}' and fsjb='{fsjb}'"
        #指定sql为删除语句以确保每次都会请求网络
        df = self.load_data(name,f"hszbl/fsjy/{code}/{fsjb}",
                                        add_fields=add_fields,
                                        sql=sql,
                                        keys=keys,date_fields=date_fields,
                                        )
        if sync_es:
            es_result = self.esLib.save(name,df,ids=keys)
            print(f"errors:{es_result["errors"]}")
        return df
    
    def get_hszbl_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="15m"):
        """获取股票代码分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszbl/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_hszbc_fsjy(self,code:str,sdt:str,edt:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="15m",sync_es=False):
        """获取股票代码某段时间分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
    
        #  http://api.mairui.club/hszbc/fsjy/股票代码(如000001)/分时级别/起始时间/结束时间/您的licence

        # 字段名称	数据类型	字段说明
        # d	string	交易时间，短分时级别格式为YYYY-MM-DD HH:MM，日线级别为yyyy-MM-dd
        # o	number	开盘价（元）
        # h	number	最高价（元）
        # l	number	最低价（元）
        # c	number	收盘价（元）
        # v	number	成交量（手）
        # e	number	成交额（元）
        # zf	number	振幅（%）
        # hs	number	换手率（%）
        # zd	number	涨跌幅（%）
        # zde	number	涨跌额（元）

        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code,"fsjb":fsjb}
        date_fields = ['d']
        keys=["mr_code","fsjb","d"]
        name = f"hszbl_fsjy_{fsjb}"
        sql = f"delete from {name} where mr_code='{mr_code}' and fsjb='{fsjb}'"
        #指定sql为删除语句以确保每次都会请求网络
        df = self.load_data(name,f"hszbl/fsjy/{code}/{fsjb}/{sdt}/{edt}",
                                        add_fields=add_fields,
                                        sql=sql,
                                        keys=keys,date_fields=date_fields,
                                        )
        if sync_es:
            es_result = self.esLib.save(name,df,ids=keys)
            print(f"errors:{es_result["errors"]}")
        return df
    def get_hszbc_ma(self,code:str,sdt:str,edt:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码某段时间分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszbc/ma/{code}/{fs}/{sdt}/{edt}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def register_router(self):
        @self.router.get("/hszb/fsjy/{code}/{fsjb}")
        async def get_hszb_fsjy(code:str,fsjb:str,req:Request):
            """获取股票代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
            try:
                df = self.get_hszb_fsjy(code,fsjb)
                if "json" in req.query_params.keys():
                    return df.to_dict(orient="records")
                df = self._prepare_df(df,req)
                content = self._to_html(df)
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
            
        @self.router.get("/hszbl/fsjy/{code}/{fsjb}")
        async def get_hszbl_fsjy(code:str,fsjb:str,req:Request):
            """获取股票代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
            try:
                df = self.get_hszbl_fsjy(code,fsjb)
                if "json" in req.query_params.keys():
                    return df.to_dict(orient="records")
                df = self._prepare_df(df,req)
                content = self._to_html(df)
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
        @self.router.get("/hszbc/fsjy/{code}/{sdt}/{edt}/{fsjb}")
        async def get_hszbc_fsjy(code:str,sdt:str,edt:str,fsjb:str,req:Request):
            """获取股票代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
            try:
                df = self.get_hszbc_fsjy(code,sdt,edt,fsjb)
                if "json" in req.query_params.keys():
                    return df.to_dict(orient="records")
                df = self._prepare_df(df,req)
                content = self._to_html(df)
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")

