from typing import Annotated, Literal
import pysnowball as ball
import tushare
import pandas as pd
import requests
import json
import os
from ylz_utils.config import Config 

from rich import print
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from ylz_utils.database.elasticsearch import ESLib
from elasticsearch_dsl import connections, Field,Document, Date, Nested, Boolean, \
    analyzer, InnerDoc, Completion, Keyword, Text,Integer,Long,Double,Float,\
    DateRange,IntegerRange,FloatRange,IpRange,Ip,Range
from ylz_utils.langchain.graph.stock_graph.state import *

class StockTools:
    stock:list = []
    def __init__(self,graphLib):
        self.graphLib = graphLib
        
        # 获取当前模块的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建 gpdm 文件的完整路径
        gpdm_file = os.path.join(current_dir, 'gpdm.json')
        with open(gpdm_file, 'r', encoding='utf-8') as f:
            self.gpdm = json.load(f)
         # 构建 zsdm 文件的完整路径
        zsdm_file = os.path.join(current_dir, 'zsdm.json')
        with open(zsdm_file, 'r', encoding='utf-8') as f:
            self.zsdm = json.load(f)
    def _get_stock_code(self,stock_name:str)->dict:
        """根据股票或指数名称获取股票/指数代码"""
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
 
        
class MairuiTools(StockTools):
    def __init__(self,graphLib):
        super().__init__(graphLib)
        self.mairui_token = Config.get('STOCK.MAIRUI.TOKEN')
        print("MAIRUI_TOKEN=",self.mairui_token)
        self.mairui_api_url = "http://api.mairui.club" 
        self._exports = [
            'get_hscp_cwzb','get_hscp_jdlr','get_hscp_jdxj',
            'get_hsmy_jddxt','get_hsmy_lscjt',
            'get_hsrl_zbdd','get_hsrl_mmwp',
            'get_hscp_.*',
            '.*'
        ]        
        self.df_hsrl_mmwp = pd.DataFrame([])       
        self.df_hsrl_zbjy = pd.DataFrame([])
    def get_hslt_list(self)->list[HSLT_LIST]:
        """获取沪深两市的公司列表"""
        res = requests.get( 
            f"{self.mairui_api_url}/hslt/list/{self.mairui_token}",
        )
        data = res.json()        
        today = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        return [HSLT_LIST(**{**item,"t":today,"dm":f"{item['jys']}{item['dm']}"}) for item in data]
    
    def get_hscp_gsjj(self, code:str)->HSCP_GSJJ:
        """获取公司基本信息和IPO基本信息"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/gsjj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return HSCP_GSJJ(**{**data,"mr_code":mr_code,"t":datetime.today().strftime("%Y-%m-%d %H:%M:%S")})
    
    def get_hscp_sszs(self, code:str):
        """获取公司所属的指数代码和名称"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/sszs/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ljgg(self, code:str):
        """获取公司历届高管成员名单"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ljgg/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ljds(self, code:str):
        """获取公司历届董事成员名单"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ljds/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ljjj(self, code:str):
        """获取公司历届监事成员名单"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ljjj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_jdlr(self, code:str)-> list[JDLR]:
        """获取公司近一年各季度利润"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/jdlr/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [JDLR(**item) for item in data]

    def get_hscp_jdxj(self, code:str) -> list[JDXJ]:
        """获取公司近一年各季度现金流"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/jdxj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [JDXJ(**item) for item in data]
    def get_hscp_cwzb(self, code:str)->list[CWZB]:
        """获取公司近一年各季度主要财务指标"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/cwzb/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [CWZB(**item) for item in data]
        #return list(map(lambda item:FinancialReport(**item),data))
    def get_hscp_sdgd(self, code:str):
        """获取公司十大股东"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/sdgd/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ltgd(self, code:str):
        """获取公司十大流通股股东"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ltgd/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_gdbh(self, code:str):
        """获取公司股东变化趋势"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/gdbh/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_jjcg(self, code:str):
        """获取公司最近500家左右的基金持股情况"""
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/jjcg/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_zlzj(self,code:str):
        """获取某个股票的每分钟主力资金走势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/zlzh/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_zjlr(self,code:str):
        """获取某个股票的近十年每天资金流入趋势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/zjlr/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_zhlrt(self,code:str):
        """获取某个股票的近10天资金流入趋势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/zhlrt/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_jddx(self,code:str):
        """获取某个股票的近十年主力阶段资金动向"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/jddx/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_jddxt(self,code:str)->list[JDDXT]:
        """获取某个股票的近十天主力阶段资金动向"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/jddxt/{code}/{self.mairui_token}",
        )
        if res.status_code==200:
            data = res.json()        
            return [JDDXT(**item) for item in data]
        else:
            return []
    def get_hsmy_lscj(self,code:str):
        """获取某个股票的近十年每天历史成交分布"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/lscj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_lscjt(self,code:str):
        """获取某个股票的近十天历史成交分布"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/lscjt/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_ssjy(self,code:str)-> SSJY:
        """获取某个股票的实时交易数据"""
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/ssjy/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [SSJY(**item) for item in data]
    def get_hszbc_fsjy(self,code:str,sdt:str,edt:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="15m"):
        """获取股票代码某段时间分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
    
        #  http://api.mairui.club/hszbc/fsjy/股票代码(如000001)/分时级别/起始时间/结束时间/您的licence
        if not hasattr(self,"df_hszbc_fsjy"):
            self.df_hszbc_fsjy = pd.DataFrame(columns=HSZBC_FSJY.model_fields.keys())
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code,"fsjb":fsjb}
        skip_condition = f"mr_code == '{mr_code}' & fsjb == '{fsjb}'"
        df = self._load_data("hszbc_fsjy.csv",f"hszbc/fsjy/{code}/{fsjb}/{sdt}/{edt}",
                                     dataframe=self.df_hszbc_fsjy,
                                     add_fields=add_fields,
                                     skip_condition=skip_condition,
                                     keys=["mr_code",'fsjb'])
        return df
    def _load_data(self,file_name:str,method_path:str,dataframe:pd.DataFrame,
                           add_fields:dict={},skip_condition:str=None,keys=['mr_code']):
        try:
            df=None
            get_df = None
            cache_df = None
            if dataframe.empty:
                try:
                    dataframe = pd.read_csv(file_name)
                    # 判断是否rload
                    print("skip_condition:",skip_condition)
                    if not skip_condition:
                        print("!!!!! ALWAYS RELOAD !!!!")
                        raise Exception("always reload")
                    cache_df = dataframe.query(skip_condition) 
                    if cache_df.empty:
                        print("!!!!! NEED RELOAD !!!!!")
                        raise Exception("need reload")
                except Exception as e:
                    try:
                        print("?????? method path:",method_path)
                        res = requests.get(f"{self.mairui_api_url}/{method_path}/{self.mairui_token}")
                        data = res.json()
                        print("!!!!!!! get data count:",len(data))
                        if isinstance(data,list):
                            data = [{**item,**add_fields} for item in data]
                        else:
                            data = [{**data,**add_fields}]
                        get_df = pd.DataFrame(data) 
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
    def get_hszg_zg(self,code:str):
        '''根据股票找相关指数、行业、概念'''
        #http://api.mairui.club/hszg/zg/股票代码(如000001)/您的licence
        if not hasattr(self,"df_hszg_zg"):
            self.df_hszg_zg = pd.DataFrame(columns=HSZG_ZG.model_fields.keys())
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"t":datetime.strftime(datetime.now(),"%Y-%m-%d 00:00:00"),"mr_code":mr_code}
        skip_condition = f"mr_code == '{mr_code}'"
        df = self._load_data("hszg_zg.csv",f"hszg/zg/{code}",
                                     dataframe=self.df_hszg_zg,
                                     add_fields=add_fields,
                                     skip_condition=skip_condition,
                                     keys=["mr_code"])
        return df
    def get_hsrl_mmwp(self,code:str,state: Annotated[NewState, InjectedState]=None)->list[HSRL_MMWP]:
        """获取某个股票的盘口交易数据,返回值没有当前股价，仅有5档买卖需求量价以及委托统计"""
        #数据更新：交易时间段每2分钟
        #请求频率：1分钟300次
        if not hasattr(self,"df_hsrl_mmwp"):
            self.df_hsrl_mmwp = pd.DataFrame(columns=HSRL_MMWP.model_fields.keys())
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        df = self._load_data("hsrl_wwmp.csv",f"hsrl/mmwp/{code}",self.df_hsrl_mmwp,
                                     add_fields = add_fields,
                                     keys=['t','mr_code'])
        #return [HSRL_MMWP(**item) for idx,item in self.df_hsrl_mmwp[self.df_hsrl_mmwp['mr_code']==mr_code].iterrows()]
        return [HSRL_MMWP(**item) for idx,item in df.iterrows()]
    
    def get_hsrl_zbjy(self,code:str):
        """获取某个股票的当天逐笔交易数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        if not hasattr(self,"df_hsrl_zbjy"):
            self.df_hsrl_zbjy = pd.DataFrame(columns=HSRL_ZBJY.model_fields.keys())
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        df = self._load_data("hsrl_zbjy.csv",f"hsrl/zbjy/{code}",self.df_hsrl_zbjy,add_fields = add_fields,keys=['d','t','mr_code'])
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
    def get_hsrl_zbdd(self,code:str):
        """获取某个股票的当天逐笔超400手的大单成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/zbdd/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hszb_fsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszb/fsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

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
    
    def get_hszbl_fsjy(self,code:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        if not hasattr(self,"df_hszbc_fsjy"):
            self.df_hszbc_fsjy = pd.DataFrame(columns=HSZBC_FSJY.model_fields.keys())
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code,"fsjb":fsjb}
        skip_condition = f"mr_code == '{mr_code}' & fsjb == '{fsjb}'"
        df = self._load_data("hszbl_fsjy.csv",
                f"hszbl/fsjy/{code}/{fsjb}",
                dataframe=self.df_hszbc_fsjy,
                add_fields = add_fields,
                skip_condition = skip_condition,
                keys=["mr_code","fsjb"])
        return df
    def get_hszbl_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
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

    def get_zs_sssj(self,code:str):
        """获取某个指数的实时交易数据"""
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/zs/sssj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_zs_lsgl(self)->list[ZS_LSGL]:
        """获取沪深两市不同涨跌幅的股票数统计"""
        #数据更新：交易时间段每2分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        res = requests.get( 
            f"{self.mairui_api_url}/zs/lsgl/{self.mairui_token}",
        )
        data = res.json()        
        return [ZS_LSGL(**{**item,"t":datetime.today().strftime("%Y-%m-%d %H:%M:%S")}) for item in [data]]

    def get_zs_fsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/zs/fsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_zs_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易的平均移动线数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/zs/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_zs_hfsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/zs/hfsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_zs_hma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/zs/hma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hitc_jrts(self)->JRTS:
        """获取今日股票、基金公告事项以及交易异动概览"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hitc/jrts/{self.mairui_token}",
        )
        data = res.json()        
        return JRTS(**data)
    def get_hitc_dzjy(self)->list[DZJY]:
        """获取上一个交易日的大宗交易数据"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hitc/dzjy/{self.mairui_token}",
        )
        data = res.json()        
        return [DZJY(**item) for item in data]
    def get_hibk_zjhhy(self)->list[ZJHHY]:
        """获取所有证监会定义的行业板块个股统计数据"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hibk/zjhhy/{self.mairui_token}",
        )
        data = res.json()      
        return [ZJHHY(**item) for item in data]
    def get_hibk_gnbk(self)->list[GNBK]:
        """获取所有概念板块个股统计数据"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hibk/gnbk/{self.mairui_token}",
        )
        data = res.json()        
        return [GNBK(**item) for item in data]
    def get_higg_jlr(self)->list[JLR]:
        """获取所有股票的资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/jlr/{self.mairui_token}",
        )
        data = res.json()        
        return [JLR(**item) for item in data]
    def get_higg_zljlr(self)->list[ZLJLR]:
        """获取所有股票的主力资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/zljlr/{self.mairui_token}",
        )
        data = res.json()        
        return [ZLJLR(**item) for item in data]
    def get_higg_shjlr(self)->list[SHJLR]:
        """获取所有股票的散户资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/shjlr/{self.mairui_token}",
        )
        data = res.json()        
        return [SHJLR(**item) for item in data]
class TushareTools(StockTools):
    def __init__(self,graphLib):
        super().__init__(graphLib)
        self.tushare_token = Config.get('STOCK.TUSHARE.TOKEN')
        self.tuapi = tushare.pro_api(self.tushare_token)
        self.tushare_api_url = "http://api.tushare.pro"

    def _tushare_query(self, api_name, fields='', **kwargs):
        req_params = {
            'api_name': api_name,
            'token': self.tushare_token,
            'params': kwargs,
            'fields': fields
        }
        res = requests.post(
            self.tushare_api_url,
            req_params
        )

        result = json.loads(res.read().decode('utf-8'))

        if result['code'] != 0:
            raise Exception(result['msg'])

        data  = result['data']
        columns = data['fields']
        items = data['items']

        return pd.DataFrame(items, columns=columns)

class SnowballTools(StockTools): 
    def __init__(self,graphLib):
        super().__init__(graphLib)
        snowball_token = Config.get('STOCK.SNOWBALL.TOKEN')
        ball.set_token(f"xq_a_token={snowball_token};") 
        print(f"snowball token:{snowball_token}")

    def quotec(self,code:str):
        '''
        查看股票的实时行情
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.quotec(ball_code)
    def pankou(self,code:str):
        '''
        查看股票的实时分笔数据，可以实时取得股票当前报价和成交信息
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        res =  ball.pankou(ball_code)
        print(type(res['data']['timestamp']),res)
        data ={ 
        "t":datetime.fromtimestamp(res['data']['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S'),
        "mr_code":mr_code,
        "vc":res['data']['diff']/100,
        "vb":res['data']['ratio'],
        "pb1":res['data']['bp1'],
        "vb1":res['data']['bc1']/100,
        "pb2":res['data']['bp2'],
        "vb2":res['data']['bc2']/100,
        "pb3":res['data']['bp3'],
        "vb3":res['data']['bc3']/100,
        "pb4":res['data']['bp4'],
        "vb4":res['data']['bc4']/100,
        "pb5":res['data']['bp5'],
        "vb5":res['data']['bc5']/100,
        "ps1":res['data']['sp1'],
        "vs1":res['data']['sc1']/100,
        "ps2":res['data']['sp2'],
        "vs2":res['data']['sc2']/100,
        "ps3":res['data']['sp3'],
        "vs3":res['data']['sc3']/100,
        "ps4":res['data']['sp4'],
        "vs4":res['data']['sc4']/100,
        "ps5":res['data']['sp5'],
        "vs5":res['data']['sc5']/100,
        }
        return [HSRL_MMWP(**item) for item in [data]]
    def capital_flow(self,code:str):
        '''
        获取当日资金流入流出数据，每分钟数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.capital_flow(ball_code)
    def capital_history(self,code:str):
        '''
        获取历史资金流入流出数据，每日数据
        输出中sum3、sum5、sum10、sum20分别代表3天、5天、10天、20天的资金流动情况
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.capital_history(ball_code)
    def earningforecast(self,code:str):
        '''
        按年度获取业绩预告数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.earningforecast(ball_code)
    def capital_assort(self,code:str):
        '''
        获取资金成交分布数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.capital_assort(ball_code)
    def blocktrans(self,code:str):
        '''
        获取大宗交易数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.blocktrans(ball_code)
    def indicator(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        按年度、季度获取业绩报表数据。
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.indicator(symbol=ball_code,is_annals=is_annals,count=count)
    def business(self,code:str,*,count:int=10):
        '''
        获取主营业务构成数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.business(symbol=ball_code,count=count)
    def top_holders(self,code:str,*,circula=1):
        '''
        获取十大股东
        code -> 股票代码
        circula -> 只获取流通股,默认为1
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.top_holders(symbol=ball_code,circula=circula)
    def main_indicator(self,code:str):
        '''
        获取主要指标
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.main_indicator(ball_code)
    def holders(self,code:str):
        '''
        获取股东人数
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.holders(ball_code)
    def org_holding_change(self,code:str):
        '''
        获取机构持仓情况
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.org_holding_change(ball_code)
    def industry_compare(self,code:str):
        '''
        获取行业对比数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.industry_compare(ball_code)
    def income(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取利润表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.income(symbol=ball_code,is_annals=is_annals,count=count)
    def balance(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取资产负债表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，如果没有指定，可以设定为10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.balance(symbol=ball_code,is_annals=is_annals,count=count)
    def cash_flow(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取现金流量表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.cash_flow(symbol=ball_code,is_annals=is_annals,count=count)
    

if __name__ == "__main__":
    from ylz_utils.langchain import LangchainLib
    from ylz_utils.langchain.graph.stock_graph import StockGraph
    import time

    Config.init('ylz_utils')
    langchainLib = LangchainLib()
    stockGraph = StockGraph(langchainLib)
    # toolLib = SnowballTools(stockGraph)
    # data  = toolLib.balance("ST易联众")
    toolLib = MairuiTools(stockGraph)

    #data1=data2=data3=data4=data5=[]
    # for i in range(30):
    #     print("index======>",i,len(data1),len(data2),len(data3),len(data4),len(data5))
    #     data1 = toolLib.get_hsrl_mmwp("中粮资本")
    #     data2 = toolLib.get_hsrl_mmwp("蒙草生态")
    #     data3 = toolLib.get_hsrl_mmwp("瑞芯微")
    #     data4 = toolLib.get_hsrl_mmwp("万达信息")
    #     data5 = toolLib.get_hsrl_mmwp("全志科技")
    #     time.sleep(60)
    #data = toolLib.get_hsrl_zbjy("万达信息")
    #data = toolLib.get_hsrl_zbjy("万达信息")
    #data = toolLib.get_zs_hfsjy("蒙草生态")
    #data = toolLib.get_zs_lsgl()
    #data = toolLib.get_hszg_zg("旗天科技")
    #data = toolLib.get_hscp_cwzb("蒙草生态")
    #data = toolLib.get_hsrl_mmwp("福日电子")
    #data = toolLib.get_hszbc_fsjy("蒙草生态","2024-08-30","2024-08-30",'5m')
    
    data = toolLib.get_hszbl_fsjy("蒙草生态","dn")
    print(type(data))
    if isinstance(data,list):
        print(len(data))
    
    esLib = ESLib(using='es')
    data["_id"] = data['mr_code'] + '_' + data['fsjb'] + '_' + data['d']
    result = esLib.drop("example_index")
    print(result)    
    result = esLib.search("example_index",{"query":{"match":{"mr_code":"sz300355"}}})
    print(result)
    # print("雪球--->")
    # lib = SnowballTools(stockGraph)
    # res = lib.pankou('福日电子')
    # print(res)
