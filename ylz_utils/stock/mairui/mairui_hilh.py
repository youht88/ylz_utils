from datetime import datetime, timedelta
from typing import Literal
import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from .mairui_base import MairuiBase

class HILH(MairuiBase):
    def __init__(self):
        super().__init__()
        self.register_router()
    def get_hilh_mrxq(self,sync_es:bool=False):
        '''今日龙虎榜概览'''
        #数据更新：每天15:30（约10分钟完成更新）
        #请求频率：1分钟20次 

        # 字段名称	数据类型	字段说明
        # t	string	日期yyyy-MM-dd
        # dpl7	string	跌幅偏离值达7%的证券，格式：dpl7:[LhbDetail,...]，LhbDetail对象说明见下表。
        # z20	string	连续三个交易日内，涨幅偏离值累计达20%的证券，格式：z20:[LhbDetail,...]，LhbDetail对象说明见下表。
        # zpl7	string	涨幅偏离值达7%的证券，格式：zpl7:[LhbDetail,...]，LhbDetail对象说明见下表。
        # h20	string	换手率达20%的证券，格式：h20:[LhbDetail,...]，LhbDetail对象说明见下表。
        # st15	string	连续三个交易日内，涨幅偏离值累计达到15%的ST证券、*ST证券和未完成股改证券，格式：st15:[LhbDetail,...]，LhbDetail对象说明见下表。
        # st12	string	连续三个交易日内，涨幅偏离值累计达到12%的ST证券、*ST证券和未完成股改证券，格式：st12:[LhbDetail,...]，LhbDetail对象说明见下表。
        # std15	string	连续三个交易日内，跌幅偏离值累计达到15%的ST证券、*ST证券和未完成股改证券，格式：std15:[LhbDetail,...]，LhbDetail对象说明见下表。
        # std12	string	连续三个交易日内，跌幅偏离值累计达到12%的ST证券、*ST证券和未完成股改证券，格式：std12:[LhbDetail,...]，LhbDetail对象说明见下表。
        # zf15	string	振幅值达15%的证券，格式：zf15:[LhbDetail,...]，LhbDetail对象说明见下表。
        # df15	string	连续三个交易日内，跌幅偏离值累计达20%的证券，格式：df15:[LhbDetail,...]，LhbDetail对象说明见下表。
        # wxz	string	无价格涨跌幅限制的证券，格式：wxz:[LhbDetail,...]，LhbDetail对象说明见下表。
        # wxztp	string	当日无价格涨跌幅限制的A股，出现异常波动停牌的股票，格式：wxztp:[LhbDetail,...]，LhbDetail对象
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['ud']
        name = f"hilh_mrxq"
        
        df = self.load_data(name,f"hilh/mrxq",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['ud'])
            print(f"errors:{es_result["errors"]}")
        return df
    
    def get_hilh_jgxw(self,days:Literal[5,10,30,60]=5,sync_es:bool=False):   
        """获取近五个交易日（按交易日期倒序）上榜个股被机构交易的总额，以及个股上榜原因。"""
        #数据更新：每天15:30（约10分钟完成更新）
        #请求频率：1分钟20次 

        # 字段名称	数据类型	字段说明
        # dm	string	股票代码
        # mc	string	股票名称
        # be	number	累积买入额(万)
        # bcount	number	买入次数
        # se	number	累积卖出额(万)
        # scount	number	卖出次数
        # ende	number	净额(万)

        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"days":days,"ud":ud}
        date_fields = ['ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}') and (days=={days})"
        keys=['dm','days','ud']
        name = f"hilh_jgxw"
        sql = f"select * from {name} where strftime('%Y-%m-%d',ud) >= '{ud}' and days={days}"
        
        df = self.load_data(name,f"hilh/jgxw/{days}",
                            add_fields=add_fields,
                            sql=sql,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','days','ud'])
            print(f"errors:{es_result["errors"]}")
        return df

    def get_hilh_xwmx(self,sync_es:bool=False):   
        """获取近五个交易日（按交易日期倒序）上榜个股被机构交易的总额，以及个股上榜原因。"""
        #数据更新：每天15:30（约10分钟完成更新）
        #请求频率：1分钟20次 

        # 字段名称	数据类型	字段说明
        # m	string	股票代码
        # mc	string	股票名称
        # t	string	交易日期yyyy-MM-dd
        # buy	number	机构席位买入额(万)
        # sell	number	机构席位卖出额(万)
        # type	number	类型

        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"hilh_xwmx"
        
        df = self.load_data(name,f"hilh/xwmx",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df
    def register_router(self):
        @self.router.get("/hilh/mrxq",response_class=HTMLResponse)
        async def get_hilh_mrxq(req:Request):
            """今日龙虎榜概览"""
            try:
                df = self.get_hilh_mrxq()
                df = self._prepare_df(df,req)
                content = df.to_html()
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
        
        @self.router.get("/hilh/jgxw/{days}",response_class=HTMLResponse)
        async def get_hilh_mrxq(days:int,req:Request):
            """今日龙虎榜概览"""
            try:
                df = self.get_hilh_jgxw(days)
                df = self._prepare_df(df,req)
                content = df.to_html()
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")
                
        @self.router.get("/hilh/xwmx",response_class=HTMLResponse)
        async def get_hilh_xwmx(req:Request):
            """获取近五个交易日（按交易日期倒序）上榜个股被机构交易的总额，以及个股上榜原因。"""
            try:
                df = self.get_hilh_xwmx()
                df = self._prepare_df(df,req)
                content = df.to_html()
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")