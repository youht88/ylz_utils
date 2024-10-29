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
        from .mairui_hszg import HSZG
        from .mairui_himk import HIMK
        from ..snowball import SnowballStock
        hizj = HIZJ()
        hszg = HSZG()            
        himk = HIMK()
        snowball = SnowballStock()
        self.router = APIRouter()

        @self.router.get("/refresh_hizj")
        async def refresh_hizj():
            funcs = [hizj.get_hizj_bk,hizj.get_hizj_zjh,hizj.get_hizj_lxlr,hizj.get_hizj_lxlc]
            for func in funcs:
                func(sync_es = True)
            return {"message":"refresh_hizj completed!"}
        @self.router.get("/bk/{code}",response_class=HTMLResponse)
        async def get_bk_stock(code,req:Request):
            o = req.query_params.get('o')
            num = int(req.query_params.get('num',0))
            bk_data = hszg.get_hszg_gg(code)
            kwargs = {
                "func": snowball.quotec_detail,
                "codes": [item['dm'] for item in bk_data],
            }
            quotec_data = self.parallel_execute(**kwargs)
            df = pd.DataFrame(quotec_data)
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            df = df.filter(
                ['t','mr_code','name','high52w','low52w','current_year_percent','last_close',
                 'current','percent','open','high','low','chg','volume','amount','volume_ratio','turnover_rate','pankou_ratio',
                 'float_shares','total_shares','float_market_capital','market_capital',
                 'eps','dividend','pe_ttm','pe_forecast','pb','pledge_ratio','navps','amplitude','current_ext','volume_ext'])
            content = df.to_html()
            return HTMLResponse(content=content)    
        @self.router.get("/hizj/bk",response_class=HTMLResponse)
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
        @self.router.get("/hizj/zjh",response_class=HTMLResponse)
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
        @self.router.get("/hizj/lxlr",response_class=HTMLResponse)
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
        
        @self.router.get("/hizj/lxlc",response_class=HTMLResponse)
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
        
        @self.router.get("/hszg/gg/{code}",response_class=HTMLResponse)
        async def get_hszg_gg(code:str,req:Request):
            o = req.query_params.get('o')
            #asc = bool(req.query_params.get('asc','False'))
            num = int(req.query_params.get('num',0))
            data = hszg.get_hszg_gg(code)
            df = pd.DataFrame(data)
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)

        @self.router.get("/hszg/zg/{code}",response_class=HTMLResponse)
        async def get_hszg_zg(code:str,req:Request):
            o = req.query_params.get('o')
            #asc = bool(req.query_params.get('asc','False'))
            num = int(req.query_params.get('num',0))
            data = hszg.get_hszg_zg(code)
            df = pd.DataFrame(data)
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)
        
        @self.router.get("/himk/ltszph",response_class=HTMLResponse)
        async def get_himk_ltszph(req:Request):
            o = req.query_params.get('o')
            magic = req.query_params.get('magic')
            num = int(req.query_params.get('num',0))
            df = himk.get_himk_ltszph()
            if magic!=None:
                df = df[df['c'].apply(self.is_magic)] 
                df = df.reset_index(drop=True)   
            if o:
                df = df.sort_values(by=o,ascending=False)
            if num:
                df = df.head(num)
            content = df.to_html()
            return HTMLResponse(content=content)
        

        return self.router
      

        
