import json
import sqlite3
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import requests
import akshare as ak

from ylz_utils.config import Config
from ylz_utils.stock.base import StockBase
from datetime import datetime,timedelta
from tqdm import tqdm

class AkshareStock(StockBase):
    def __init__(self):
        super().__init__()
        self.register_router()
    def current(self):
        df = ak.stock_zh_a_spot_em()
        return df
    def daily_refresh(self,sdate='20220101'):
        daily_db=sqlite3.connect("daily.db")
        codes_info = self._get_bk_codes("hs_a")
        print("total codes:",len(codes_info))
        error_code_info=[]
        error_msg=""
        with tqdm(total=len(codes_info)) as pbar:
            for code_info in codes_info:
                code = code_info['dm']
                mc = code_info['mc']
                today = datetime.today().strftime("%Y%m%d")
                new_sdate = sdate
                try:
                    try:
                        max_date=daily_db.execute(f"select max(d) from daily where code='{code}'").fetchall()[0][0]
                        if max_date:
                            max_date = datetime.strptime(max_date,"%Y-%m-%d %H:%M:%S") 
                            start_date = datetime.strptime(sdate,"%Y%m%d")
                            if max_date >= start_date:
                                new_sdate = datetime.strftime(max_date + timedelta(days=1),"%Y%m%d")
                    except Exception as e:
                        print(code,mc,e)
                        error_msg="no_such_table"
                    finally:
                        if new_sdate<=today:
                            df=ak.stock_zh_a_hist(code,start_date=new_sdate)
                            if not df.empty:
                                df=df.rename(columns={'日期':'d','股票代码':'code','开盘':'o','最高':'h','最低':'l','收盘':'c','成交量':'v','成交额':'e','振幅':'zf','涨跌幅':'zd','涨跌额':'zde','换手率':'hs'})
                                df['mc']=mc
                                df['d'] = pd.to_datetime(df['d'],format='%Y-%m-%d')
                                if error_msg=="no_such_table":
                                    error_msg=""
                                    df.to_sql("daily",if_exists="replace",index=False,con=daily_db)
                                    daily_db.execute("create unique index daily_index on daily(code,d)")
                                else:
                                    df.to_sql("daily",if_exists="append",index=False,con=daily_db)
                        else:
                            print(f"no need fetch {new_sdate} of {code}-{mc}")
                except Exception as e:
                    error_code_info.append(f"{code}-{mc}")
                pbar.update(1)
        return error_code_info
    def fx1(self):
        daily_db = sqlite3.connect("daily.db")
        df = pd.read_sql("select * from daily",daily_db)
        dfg = df.groupby(by="code")
        return dfg.get_group("300096")
    def daily_pro(self):
        daily_db=sqlite3.connect("daily.db")
        daily_pro_db=sqlite3.connect("daily_pro.db")
        table = "daily"
        codes_info = self._get_bk_codes("hs_a")
        #codes_info = codes_info[:10]
        print("total codes:",len(codes_info))
        with tqdm(total=len(codes_info),desc="进度") as pbar:
            days=15
            error_code_info=[]
            error_msg=""
            sdate = '2022-01-01'
            for code_info in codes_info:
                code = code_info['dm']
                mc = code_info['mc']
                new_sdate = sdate
                max_date=None
                try:
                    max_date=daily_pro_db.execute(f"select max(d) from daily where code='{code}'").fetchall()[0][0]
                    if max_date:
                        d_max_date = datetime.strptime(max_date,"%Y-%m-%d %H:%M:%S") 
                        d_start_date = datetime.strptime(sdate,"%Y-%m-%d")
                        if d_max_date >= d_start_date:
                            new_sdate = datetime.strftime(d_max_date + timedelta(days=-100),"%Y-%m-%d")
                except Exception as e:
                    print(code,mc,e)
                    error_msg="no_such_table"
                    raise e
                try:
                    origin_df = pd.read_sql(f"select * from {table} where code='{code}' and d >= '{new_sdate}'",daily_db)
                    if max_date and max_date >= origin_df['d'].iloc[-1][:10]:
                        raise Exception("no deed because of data exists!")  
                    df = origin_df.reset_index() 
                    df_add_cols={}
                    for col in ['zd']:
                        for idx in range(days):
                            t=idx+1
                            df_add_cols[f'p{col}_{t}']=df[col].shift(-t)

                    df_add_cols['ma5c']=df['c'].rolling(window=5).apply(lambda x:x.mean())        
                    df_add_cols['ma5v']=df['v'].rolling(window=5).apply(lambda x:x.mean())        
                    df_add_cols['ma5hs']=df['hs'].rolling(window=5).apply(lambda x:x.mean())        
                    df_add_cols['ma5zf']=df['zf'].rolling(window=5).apply(lambda x:x.mean())        
                    df_add_cols['ma10c']=df['c'].rolling(window=10).apply(lambda x:x.mean())        
                    df_add_cols['ma10v']=df['v'].rolling(window=10).apply(lambda x:x.mean())        
                    df_add_cols['ma10hs']=df['hs'].rolling(window=10).apply(lambda x:x.mean())        
                    df_add_cols['ma10zf']=df['zf'].rolling(window=10).apply(lambda x:x.mean())        
                    df_add_cols['ma20c']=df['c'].rolling(window=20).apply(lambda x:x.mean())        
                    df_add_cols['ma20v']=df['v'].rolling(window=20).apply(lambda x:x.mean())        
                    df_add_cols['ma20hs']=df['hs'].rolling(window=20).apply(lambda x:x.mean())        
                    df_add_cols['ma20zf']=df['zf'].rolling(window=20).apply(lambda x:x.mean())        
                    df_add_cols['ma60c']=df['c'].rolling(window=60).apply(lambda x:x.mean())        
                    df_add_cols['ma60v']=df['v'].rolling(window=60).apply(lambda x:x.mean())        
                    df_add_cols['ma60hs']=df['hs'].rolling(window=60).apply(lambda x:x.mean())        
                    df_add_cols['ma60zf']=df['zf'].rolling(window=60).apply(lambda x:x.mean())        
                    
                    for col in ['o','c','v','hs','zf']:
                        for idx in range(days):
                            t=idx+1
                            df_add_cols[f'{col}{t}']=df[col].shift(t)
                    df_cols = pd.concat(list(df_add_cols.values()), axis=1, keys=list(df_add_cols.keys()))
                    df = pd.concat([origin_df,df_cols],axis=1)
                    if max_date:
                        new_sdate = datetime.strftime(d_max_date + timedelta(days=1),"%Y-%m-%d")
                        df = df[df['d']>=new_sdate]
                    if error_msg=="no_such_table":
                        error_msg=""
                        df.to_sql("daily",if_exists="replace",index=False,con=daily_pro_db)
                        daily_db.execute("create unique index daily_index on daily(code,d)")
                    else:
                        df.to_sql("daily",if_exists="append",index=False,con=daily_pro_db)
                except Exception as e:
                    error_code_info.append(f"{code}-{mc},{e}")
                pbar.update(1)
        return error_code_info
    
    def register_router(self):
        @self.router.get("/daily/refresh")
        async def daily_refresh(req:Request):
            """更新日线数据"""
            sdate = req.query_params.get('sdate')
            error_code_info=[]
            if sdate:
                sdate=sdate.replace('-','')
                error_code_info=self.daily_refresh(sdate)
            else:
                error_code_info=self.daily_refresh()
            return {"message":f"daily refresh已完成,error_code_info={error_code_info}"}
        @self.router.get("/daily/pro")
        async def daily_pro(req:Request):
            """更新日线增强数据"""
            error_code_info=self.daily_pro()
            return {"message":f"daily_pro已完成,error_code_info={error_code_info}"}
       
        @self.router.get("/fx1")
        async def fx1(req:Request):
            """fx1"""
            try:
                df = self.fx1()
                df = self._prepare_df(df,req)
                content = self._to_html(df)
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
        @self.router.get("/current")
        async def current(req:Request):
            """获取当前行情数据"""
            try:
                df = self.current()
                df = self._prepare_df(df,req)
                content = self._to_html(df)
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
        @self.router.get("/lxslxd")
        async def lxslxd(req:Request):
            """分析连续缩量下跌信息"""
            try:
                daily_pro_db=sqlite3.connect("daily_pro.db")
                table = "daily"
                codes_info = self._get_bk_codes("hs_a")
                #codes_info = codes_info[:100]
                print("total codes:",len(codes_info))
                error_code_info=[]
                error_msg=""
                
                days = req.query_params.get('days')
                if days:
                    days=int(days)
                else:
                    days=5
                with tqdm(total=len(codes_info),desc="进度") as pbar:
                    dfs=[]
                    for code_info in codes_info:
                        code = code_info['dm']
                        df = pd.read_sql(f"select * from {table} where code='{code}'",daily_pro_db)
                        cond=pd.Series([True] * len(df))
                        for field in ['c','v']:
                            for idx in range(days):
                                idx_str = '' if idx==0 else str(idx)
                                # if (idx== 0 or idx==1) and field=='v':
                                #     cond1= (df[f'{field}{idx_str}'] / df[f'{field}{idx+1}']) > 1.3
                                #     cond = cond & cond1
                                # elif (idx== 0 or idx==1) and field=='c':
                                #     cond1= (df[f'{field}{idx_str}'] / df[f'{field}{idx+1}']) > 1.1
                                #     cond = cond & cond1
                                if field=='c':
                                    cond1= df[f'{field}{idx_str}'].lt(df[f'{field}{idx+1}'])
                                    if field=='c':
                                        cond2=df[f'{field}{idx_str}'].lt(df[f'o{idx+1}'])
                                        cond = cond & (cond1 & cond2)
                                    else:    
                                        cond = cond & cond1
                                if field=='v':
                                    cond1= df[f'{field}{idx_str}'].gt(df[f'{field}{idx+1}'])
                                    cond = cond & cond1
                                        
                        for idx in range(days):
                            idx_str = '' if idx==0 else str(idx)
                            cond = cond & df[f'hs{idx_str}'].gt(3) & df[f'hs{idx_str}'].lt(10)
                        df_new = df[cond]
                        dfs.append(df_new)
                        pbar.update(1)
                df_concat = pd.concat(dfs,ignore_index=True)
                print("df_concat:",len(df_concat))
                df = self._prepare_df(df_concat,req)
                content = self._to_html(df,columns=['code','mc'])
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
        @self.router.get("/lxxd")
        async def lxxd(req:Request):
            """分析连续下跌信息"""
            try:
                daily_db=sqlite3.connect("daily.db")
                table = "daily"
                codes_info = self._get_bk_codes("hs_a")
                #codes_info = codes_info[:100]
                print("total codes:",len(codes_info))
                error_code_info=[]
                error_msg=""
                
                days = req.query_params.get('days')
                if days:
                    days=int(days)
                else:
                    days=5
                sdate = req.query_params.get('sdate')
                if not sdate:
                    sdate = '2022-01-01'
                with tqdm(total=len(codes_info),desc="进度") as pbar:
                    dfs=[]
                    for code_info in codes_info:
                        code = code_info['dm']
                        origin_df = pd.read_sql(f"select * from {table} where code='{code}' and d >= '{sdate}'",daily_db)
                        df = origin_df.reset_index() 
                        for idx in range(days):
                            t=idx+1
                            df[f'pzd_{t}']=df['zd'].shift(-t)
                        for idx in range(days):
                            t=idx+1
                            df[f'o{t}']=df['o'].shift(t)
                            df[f'c{t}']=df['c'].shift(t)
                            df[f'v{t}']=df['v'].shift(t)
                            df[f'e{t}']=df['e'].shift(t)
                            df[f'po{t}']=df['o'].shift(-t)
                            df[f'pc{t}']=df['c'].shift(-t)
                        cond=pd.Series([True] * len(df))
                        for field in ['c']:
                            for idx in range(days):
                                if idx==0:
                                    cond1 = df[f'{field}'] < df[f'{field}{idx+1}']
                                    if field=='c':
                                        cond2=df[f'{field}'] < df[f'o{idx+1}']
                                        cond = cond & (cond1 & cond2)
                                    else:    
                                        cond = cond & cond1
                                else:
                                    cond1= df[f'{field}{idx}'].lt(df[f'{field}{idx+1}'])
                                    if field=='c':
                                        cond2=df[f'{field}{idx}'].lt(df[f'o{idx+1}'])
                                        cond = cond & (cond1 & cond2)
                                    else:    
                                        cond = cond & cond1
                        df_new = df[cond]
                        dfs.append(df_new)
                        pbar.update(1)
                df_concat = pd.concat(dfs,ignore_index=True)
                print("df_concat:",len(df_concat))
                df = self._prepare_df(df_concat,req)
                content = self._to_html(df,columns=['code','mc'])
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")

    
