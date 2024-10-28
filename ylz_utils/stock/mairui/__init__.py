from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import requests

from ylz_utils.stock import StockLib
from rich.progress import Progress

class MairuiStock(StockLib):
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
    
    def setup_router(self):
        from .mairui_hizj import HIZJ
        hizj = HIZJ()
            
        self.router = APIRouter()
        @self.router.get("/refresh_hizj")
        async def refresh_hizj():
            progress = Progress()
            task = progress.add_task("[green]Refreshing...", total=100)
            funcs = [hizj.get_hizj_bk,hizj.get_hizj_zjh,hizj.get_hizj_lxlr,hizj.get_hizj_lxlc]
            for func in funcs:
                func(sync_es = True)
                progress.update(task, advance=1)
                progress.refresh()
            progress.complete(task)
            return {"message":"refresh_hizj completed!"}
        @self.router.get("/get_hizj_bk",response_class=HTMLResponse)
        async def get_hizj_bk(req:Request):
            o = req.query_params.get('o')
            #asc = bool(req.query_params.get('asc','False'))
            num = int(req.query_params.get('num',0))
            df = hizj.get_hizj_bk()
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)
        @self.router.get("/get_hizj_zjh",response_class=HTMLResponse)
        async def get_hizj_zjh(req:Request):
            o = req.query_params.get('o')
            #asc = bool(req.query_params.get('asc','False'))
            num = int(req.query_params.get('num',0))
            df = hizj.get_hizj_zjh()
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)
        @self.router.get("/get_hizj_lxlr",response_class=HTMLResponse)
        async def get_hizj_lxlr(req:Request):
            o = req.query_params.get('o')
            #asc = bool(req.query_params.get('asc','False'))
            num = int(req.query_params.get('num',0))
            df = hizj.get_hizj_lxlr()
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)
        
        @self.router.get("/get_hizj_lxlc",response_class=HTMLResponse)
        async def get_hizj_lxlc(req:Request):
            o = req.query_params.get('o')
            #asc = bool(req.query_params.get('asc','False'))
            num = int(req.query_params.get('num',0))
            df = hizj.get_hizj_lxlc()
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)
        
        return self.router
        

        
