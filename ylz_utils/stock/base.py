import sqlite3
from fastapi.responses import HTMLResponse
import pandas as pd
import requests
from datetime import datetime
import logging
import re
import requests
from rich import print

from ylz_utils.config import Config 
from ylz_utils.database.elasticsearch import ESLib

import concurrent.futures

from fastapi import FastAPI, HTTPException,Request,Response,APIRouter

class StockBase:
    stock:list = []
    def __init__(self):
        self.esLib = ESLib(using='es')
        # 获取当前模块的目录
        self.gpdm = None
        self.bkdm = None
        self.mairui_token = Config.get('STOCK.MAIRUI.TOKEN')
        self.mairui_api_url = "http://api.mairui.club" 
        self.router = APIRouter()
        stock_db_name = Config().get('STOCK.DB_NAME','stock.db')
        self.sqlite = sqlite3.connect(stock_db_name)
    # 定义一个执行函数的方法k
    def parallel_execute(self,**kwargs):
        func = kwargs.pop("func")
        codes = kwargs.get("codes")
        sync_sqlite = kwargs.get("sync_sqlite")
        if func:
            logging.info(f"run {func.__name__} as {datetime.now()}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if sync_sqlite:
                    kwargs.pop("sync_sqlite")
                if codes:
                    kwargs.pop("codes")
                    futures = [executor.submit(func,code,**kwargs) for code in codes]
                else:
                    futures = [executor.submit(func,**kwargs)]
                results = [future.result() for future in futures]
                print("results count:",len(results))
                if sync_sqlite:
                    db_name = sync_sqlite.get('db_name')
                    table_name = sync_sqlite.get('table_name')
                    if db_name and table_name:
                        df = pd.DataFrame(results)
                        df.to_sql(table_name,index=False,if_exists="append",con=sqlite3.connect(db_name))
                return results
    def _get_bk_code(self,bk_name:str)->dict:
        if not self.bkdm:
            res = requests.get(f"{self.mairui_api_url}/hszg/list/{self.mairui_token}")
            self.bkdm = res.json()
        code_info = list(filter(lambda item:item['code']==bk_name,self.bkdm))
        if code_info:
            if len(code_info)>1:
                print("code_info",code_info)
                raise Exception('板块代码不唯一，请重新配置!')
            return code_info[0]
        else:
            code_info = list(filter(lambda item:item['name'].find(bk_name)>=0,self.bkdm)) 
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
                zs_info = list(filter(lambda item:item["mc"]==stock_name.upper(),self.bkdm))
                if zs_info:
                    mr_code = zs_info[0]['dm']
                    jys = zs_info[0]['jys']
                    name = zs_info[0]['mc']
                    code = mr_code[2:]
                    ts_code = f"{code}.{jys.upper()}"
                    ball_code = f"{jys.upper()}{code}"
                    return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
    def _prepare_df(self,df:pd.DataFrame,req:Request):
        condition = [item for item in req.query_params.items() if '@' in item[0]]
        order = req.query_params.get('o')
        limit = int(req.query_params.get('n',0))
        number_pattern = r'^-?\d*\.?\d+$'
        if condition:
            for item in condition:
                key = item[0].replace('@','')
                if '[' in item[1] and ']' in item[1]:
                    values = item[1].replace('[','').replace(']','').split(',')[:2]
                    keys = key.split('.')
                    key = keys[0]
                    key_func = keys[1] if len(keys)>1 else None 
                    if pd.api.types.is_datetime64_any_dtype(df[key]):
                        if key_func=='date':
                            if values[0]!='' and values[1]!='':
                                print(f"{key} date between {values[0]} and {values[1]}")
                                start = pd.to_datetime(values[0]).date()
                                end = pd.to_datetime(values[1]).date()
                                df = df[(df[key].dt.date>=start) & (df[key].dt.date<=end)]
                            elif values[0]!='' and values[1]=='':
                                print(f"{key} date >= {values[0]}")
                                start = pd.to_datetime(values[0]).date()
                                df = df[(df[key].dt.date>=start)]
                            elif values[0]=='' and values[1]!='':
                                print(f"{key} date <= {values[1]}")
                                end = pd.to_datetime(values[1]).date()
                                df = df[(df[key].dt.date<=end)]
                            else:
                                print(f"{key} date == {values[0]}")
                                start = pd.to_datetime(values[0]).date()
                                df = df[(df[key].dt.date==start)]
                        elif key_func=='time':
                            if values[0]!='' and values[1]!='':
                                print(f"{key} time between {values[0]} and {values[1]}")
                                start = pd.to_datetime(values[0]).time()
                                end = pd.to_datetime(values[1]).time()
                                df = df[(df[key].dt.time>=start) & (df[key].dt.time<=end)]
                            elif values[0]!='' and values[1]=='':
                                print(f"{key} time >= {values[0]}")
                                start = pd.to_datetime(values[0]).time()
                                df = df[(df[key].dt.time>=start)]
                            elif values[0]=='' and values[1]!='':
                                print(f"{key} time <= {values[1]}")
                                end = pd.to_datetime(values[1]).time()
                                df = df[(df[key].dt.time<=end)]
                            else:
                                print(f"{key} time == {values[0]}")
                                start = pd.to_datetime(values[0]).time()
                                df = df[(df[key].dt.time==start)]
                        else:
                            if values[0]!='' and values[1]!='':
                                print(f"{key} datetime between {values[0]} and {values[1]}")
                                start = pd.to_datetime(values[0])
                                end = pd.to_datetime(values[1])
                                df = df[(df[key]>=start) & (df[key]<=end)]                      
                            elif values[0]!='' and values[1]=='':
                                print(f"{key} datetime >= {values[0]}")
                                start = pd.to_datetime(values[0])
                                df = df[(df[key]>=start)]
                            elif values[0]=='' and values[1]!='':
                                print(f"{key} datetime <= {values[1]}")
                                end = pd.to_datetime(values[1])
                                df = df[(df[key]<=end)]
                            else:
                                print(f"{key} datetime == {values[0]}")
                                start = pd.to_datetime(values[0])
                                df = df[(df[key].dt.time==start)]
                    else:
                        if re.match(number_pattern, values[0]) and re.match(number_pattern, values[1]):
                            print(f"{key} between {float(values[0])} and {float(values[1])}")
                            df = df[(df[key]>=float(values[0])) & (df[key]<=float(values[1]))]
                        elif re.match(number_pattern, values[0]) and values[1]=='':
                            print(f"{key} >= {float(values[0])}")
                            df = df[df[key] >= float(values[0])]
                        elif values[0]=='' and re.match(number_pattern, values[1]):
                            print(f"{key} <= {float(values[1])}")
                            df = df[df[key] <= float(values[1])]
                        elif re.match(number_pattern, values[0]):
                            print(f"{key} = {float(values[0])}")
                            df = df[df[key]==float(values[0])]
                        else:
                            print(f"{key} = {values[0]}")
                            df = df[df[key]==values[0]]
                else:
                    keys = key.split('.')
                    key = keys[0]
                    key_func = keys[1] if len(keys)>1 else None 
                    if key_func == 'regex':
                        values = item[1].split(',')
                        print(f"{key} match {values[0]}")
                        df = df[df[key].str.contains(values[0],regex=True)]
                    else:
                        # 包含字符串或不包含字符串
                        values = item[1].split(',')
                        filter = pd.Series([False] * len(df))
                        for value in values:
                            if value.startswith("!"):
                                print(f"{key} not contains {value[1:]}")
                                filter = filter | ~df[key].str.contains(value[1:])
                            else:
                                print(f"{key} contains {value}")
                                filter = filter | df[key].str.contains(value)
                        df = df[filter]
        if order:
            df = self._parse_order_express(df,order) 
        if limit:
            df = df.head(limit)
        return df
    def _parse_order_express(self,df:pd.DataFrame,order:str)->pd.DataFrame:
        express = re.findall(r'(add|sub|mul|div|avg)\((.*)\)',order)
        if not express:
            if '(' in order or ')' in order:
                raise Exception(f"{order}表达式不正确")
            else:
                express=[order]
        else:
            express = [express[0][0]] + express[0][1].split(',')
        number_pattern = r'^-?\d*\.?\d+$'
        if express[0]=='add' and len(express)>1:
            order = "_order"
            args = [float(arg) if re.match(number_pattern,arg) else df[arg] for arg in express[1:]]
            df["_order"] = args[0]
            for item in args[1:]:
                df["_order"] = df["_order"] + item
        elif express[0]=='sub' and len(express)==3:
            order = "_order"
            args = [float(arg) if re.match(number_pattern,arg) else df[arg] for arg in express[1:]]
            df["_order"] = args[0] - args[1]
        elif express[0]=='mul' and len(express)>1:
            order = "_order"
            args = [float(arg) if re.match(number_pattern,arg) else df[arg] for arg in express[1:]]
            df["_order"] = args[0]
            for item in args[1:]:
                df["_order"] = df["_order"] * item
        elif express[0]=='div' and len(express)==3:
            order = "_order"
            args = [float(arg) if re.match(number_pattern,arg) else df[arg] for arg in express[1:]]
            df["_order"] = args[0] / args[1]
        elif express[0]=='avg' and len(express)>1:
            order = "_order"
            args = [float(arg) if re.match(number_pattern,arg) else df[arg] for arg in express[1:]]
            df["_order"] = args[0]
            for item in args[1:]:
                df["_order"] = df["_order"] + item
            df["_order"] = df["_order"]/len(args)
        elif len(express)==1:
            pass
        else:
            raise Exception(f"{order}表达式不正确")
        df = df.sort_values(by=order,ascending=False)
        return df
    def _to_html(self,df:pd.DataFrame,columns:list[str]=[]):
        df_state = df.describe()
        df_state.loc['sum']=df[df.select_dtypes(include=['int', 'float']).columns].sum()
        state = df_state.to_html()
        data = df.to_html()
        col_text = ''
        for col in columns:
            col_text +='<p>' + ','.join(df[col].tolist()) + '</p>\n'
        content = (
            "<html>\n"
            f"{state}\n"
            f"{data}\n"
            f"{col_text}\n"
            "</html>\n"
        )
        return content
    
    def register_router(self):
        from .mairui.mairui_hszg import HSZG
        from .snowball import SnowballStock
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
                content = self._to_html(df)
                return HTMLResponse(content=content) 
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")  
            
        @self.router.get("/es/sql",response_class=HTMLResponse)
        async def es_sql(req:Request):
            sql = req.query_params.get("q")
            data = self.esLib.sql(sql)
            df = pd.DataFrame(data['rows'],columns=[item['name'] for item in data['columns']])
            content = self._to_html(df)
            return HTMLResponse(content=content)
        @self.router.get("/es/drop/{table}")
        async def es_drop_table(table:str):
            res = self.esLib.drop(table)
            return res
        
        @self.router.get("/sqlite/sql",response_class=HTMLResponse)
        async def sqlite_sql(req:Request):
            sql = req.query_params.get("q")
            df = pd.read_sql(sql,con=self.sqlite)
            content = self._to_html(df)
            return HTMLResponse(content=content)
        
        @self.router.get("/sqlite/drop/{table}")
        async def sqlite_drop_table(table:str):
            res = self.esLib.drop(table)
            return res
