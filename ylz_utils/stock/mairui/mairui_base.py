import re
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import requests
import sqlite3
from ylz_utils.stock.base import StockBase

class MairuiBase(StockBase):
    def __init__(self):
        super().__init__()
        self._exports = [
            'get_hscp_cwzb','get_hscp_jdlr','get_hscp_jdxj',
            'get_hsmy_jddxt','get_hsmy_lscjt',
            'get_hsrl_zbdd','get_hsrl_mmwp',
            'get_hscp_.*',
            '.*'
        ] 
        self.dataframe_map:dict[str,pd.DataFrame]={} 
    def is_magic(self,value):
        value_str = str(value)
        if '.' in value_str:
            integer_part = value_str.split('.')[0]
            decimal_part = value_str.split('.')[1]
            if len(decimal_part) >= 2 :
                if decimal_part[0] == decimal_part[1]:
                    return True
                if integer_part[-1] == decimal_part[1]:
                    return True
        return False
    def load_data(self,name:str,method_path:str,sql:str=None,
                           add_fields:dict={},keys=[],date_fields=[])->pd.DataFrame:
        # dataframe 传入的类实例变量，用于多次检索时的内存cache
        # add_fields 需要增加的字段，例如 add_fields = {"mr_code":"abcd"}
        # keys 判断是否重复数据的字段名数组,例如 keys=["mr_code"]
        # date_fields 指定日期类型的字段名
        # sql 从sqlite取数据的sql语句
        try:
            file_name = f"{name}.db"
            df_name = f"df_{name}"             
            if not isinstance(self.dataframe_map.get(df_name),pd.DataFrame):
                self.dataframe_map[df_name] = pd.DataFrame([])
            dataframe = self.dataframe_map[df_name]
            df=None
            get_df = None
            cache_df = None
            if dataframe.empty:
                try:
                    conn = sqlite3.connect(f"{file_name}")
                    if sql:
                        print("sql=",sql)
                        dataframe = pd.read_sql(sql,con=conn)
                    else:
                        dataframe = pd.read_sql(f"select * from {name}",con=conn)
                    for col in date_fields:
                        dataframe[col] = pd.to_datetime(dataframe[col])
                    # 判断是否rload
                    if not sql:
                        print("!!!!! ALWAYS LOAD FROM NETWORK !!!!")
                        raise Exception("always reload")
                    else:
                        cache_df = dataframe
                    if cache_df.empty:
                        print("!!!!! NEED LOAD FROM NETWORK !!!!!")
                        raise Exception("need load from network")
                except Exception as e:
                    print("start retrieve data from network...",e)
                    try:
                        print(f"{self.mairui_api_url}/{method_path}/{self.mairui_token}")
                        res = requests.get(f"{self.mairui_api_url}/{method_path}/{self.mairui_token}")
                        data = res.json()
                        if isinstance(data,list):
                            data = [{**item,**add_fields} for item in data]
                        else:
                            data = [{**data,**add_fields}]
                        get_df = pd.DataFrame(data)
                        for col in date_fields:
                            get_df[col] = pd.to_datetime(get_df[col]) 
                        if dataframe.empty:
                            dataframe = get_df
                            index_sql = f'CREATE UNIQUE INDEX idx_{name}_{"_".join(keys)} ON {name} ({",".join(keys)});'
                            dataframe.to_sql(name,index=False,if_exists="append",con=conn)                       
                            conn.execute(index_sql)
                    except Exception as e:
                        print("network error:",e)
                        raise  
            if not (get_df is None):
                print("get_df.count=",get_df.shape)                  
            if not (cache_df is None):
                print("cache_df.count=",cache_df.shape)
            if not (dataframe is None):
                print("dataframe.count=",dataframe.shape)                  
            if not dataframe.empty:
                if not (get_df is None):
                    condition = pd.Series([True] * len(dataframe))
                    # 根据字段名列表构建动态筛选条件
                    for k in keys:
                        if k in get_df.columns:
                            condition = condition & (dataframe[k] == get_df[k][0])
                    # 应用筛选条件
                    find_row:pd.DataFrame = dataframe[condition]
                    if find_row.empty:
                        dataframe = pd.concat([dataframe,get_df], ignore_index=True)
                        dataframe.to_sql(name,index=False,if_exists="append",con=conn)
                    df = get_df
                elif not (cache_df is None):
                    df = cache_df
        except Exception as e:
            raise Exception(f"error on load_data,the error is :{e}")
        return df

    def _load_data(self,name:str,method_path:str,
                           add_fields:dict={},skip_condition:str=None,keys=[],date_fields=[])->pd.DataFrame:
        # dataframe 传入的类实例变量，用于多次检索时的内存cache
        # add_fields 需要增加的字段，例如 add_fields = {"mr_code":"abcd"}
        # skip_condition 过滤的字符串，如果过滤条件下没有数据则需要从network中获取
        # keys 判断是否重复数据的字段名数组,例如 keys=["mr_code"]
        # date_fields 指定日期类型的字段名
        try:
            file_name = f"{name}.csv"
            df_name = f"df_{name}"             
            if not isinstance(self.dataframe_map.get(df_name),pd.DataFrame):
                self.dataframe_map[df_name] = pd.DataFrame([])
            dataframe = self.dataframe_map[df_name]
            df=None
            get_df = None
            cache_df = None
            if dataframe.empty:
                try:
                    dataframe = pd.read_csv(file_name,parse_dates=date_fields)
                    for col in date_fields:
                        dataframe[col] = pd.to_datetime(dataframe[col])
                    # 判断是否rload
                    print("skip_condition:",skip_condition)
                    if not skip_condition:
                        print("!!!!! ALWAYS LOAD FROM NETWORK !!!!")
                        raise Exception("always reload")
                    cache_df = dataframe.query(skip_condition) 
                    if cache_df.empty:
                        print("!!!!! NEED LOAD FROM NETWORK !!!!!")
                        raise Exception("need load from network")
                except Exception as e:
                    print("start retrieve data from network...",e)
                    try:
                        res = requests.get(f"{self.mairui_api_url}/{method_path}/{self.mairui_token}")
                        data = res.json()
                        if isinstance(data,list):
                            data = [{**item,**add_fields} for item in data]
                        else:
                            data = [{**data,**add_fields}]
                        get_df = pd.DataFrame(data)
                        for col in date_fields:
                            get_df[col] = pd.to_datetime(get_df[col]) 
                        if dataframe.empty:
                            dataframe = get_df
                            dataframe.to_csv(file_name,index=False)                       
                    except Exception as e:
                        print("network error:",e)
                        raise  
            if not (get_df is None):
                print("get_df.count=",get_df.shape)                  
            if not (cache_df is None):
                print("cache_df.count=",cache_df.shape)
            if not (dataframe is None):
                print("dataframe.count=",dataframe.shape)                  
            if not dataframe.empty:
                if not (get_df is None):
                    condition = pd.Series([True] * len(dataframe))
                    # 根据字段名列表构建动态筛选条件
                    for k in keys:
                        if k in get_df.columns:
                            condition = condition & (dataframe[k] == get_df[k][0])
                    # 应用筛选条件
                    find_row:pd.DataFrame = dataframe[condition]
                    if find_row.empty:
                        dataframe = pd.concat([dataframe,get_df], ignore_index=True)
                        dataframe.to_csv(file_name,index=False)
                    df = get_df
                elif not (cache_df is None):
                    df = cache_df
        except Exception as e:
            raise Exception(f"error on _load_data_by_code,the error is :{e}")
        return df
    def register_router(self):
        from .mairui_hszg import HSZG
        from ..snowball import SnowballStock
        hszg = HSZG()          
        snowball = SnowballStock()

        @self.router.get("/bk/{code}",response_class=HTMLResponse)
        async def get_bk_stock(code,req:Request):
            '''获取板块中权重股票的实时交易情况'''
            try:
                bk_data = hszg.get_hszg_gg(code)
                kwargs = {
                    "func": snowball.quotec_detail,
                    "codes": [item['dm'] for item in bk_data],
                }
                quotec_data = self.parallel_execute(**kwargs)
                df = pd.DataFrame(quotec_data)
                df = df.filter(
                    ['t','mr_code','name','high52w','low52w','current_year_percent','last_close',
                    'current','percent','open','high','low','chg','volume','amount','volume_ratio','turnover_rate','pankou_ratio',
                    'float_shares','total_shares','float_market_capital','market_capital',
                    'eps','dividend','pe_ttm','pe_forecast','pb','pledge_ratio','navps','amplitude','current_ext','volume_ext'])
                df = self._prepare_df(df,req)
                content = df.to_html()
                return HTMLResponse(content=content) 
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")  

