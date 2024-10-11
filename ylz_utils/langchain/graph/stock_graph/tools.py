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
    def _get_stock_code(self,stock_name:str):
        """根据股票或指数名称获取股票/指数代码"""
        stock_info = list(filter(lambda item:item["name"]==stock_name.upper(),self.gpdm))
        if stock_info:
            stock_code = stock_info[0]['symbol']
            return stock_code
        else:
            zhishu_info = list(filter(lambda item:item["mc"]==stock_name.upper(),self.zsdm))
            if zhishu_info:
                zhishu_code = zhishu_info[0]['dm']
                return zhishu_code
        return stock_name
        
class MairuiTools(StockTools):
    def __init__(self,graphLib):
        super().__init__(graphLib)
        self.mairui_token = Config.get('STOCK.MAIRUI.TOKEN')
        self.mairui_api_url = "http://api.mairui.club" 
        self._exports = [
            'get_hscp_cwzb','get_hscp_jdlr','get_hscp_jdxj',
            'get_hsmy_jddxt','get_hsmy_lscjt',
            'get_hsrl_zbdd','get_hsrl_mmwp'
        ] 
    def __call__(self,state:NewState,config:RunnableConfig):
        pass      
    def get_hscp_gsjj(self, code:str)->CompanyInfo:
        """获取公司基本信息和IPO基本信息"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/gsjj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return CompanyInfo(**data)
    
    def get_hscp_sszs(self, code:str):
        """获取公司所属的指数代码和名称"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/sszs/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ljgg(self, code:str):
        """获取公司历届高管成员名单"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ljgg/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ljds(self, code:str):
        """获取公司历届董事成员名单"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ljds/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ljjj(self, code:str):
        """获取公司历届监事成员名单"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ljjj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_jdlr(self, code:str)-> list[JDLR]:
        """获取公司近一年各季度利润"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/jdlr/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [JDLR(**item) for item in data]

    def get_hscp_jdxj(self, code:str) -> list[JDXJ]:
        """获取公司近一年各季度现金流"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/jdxj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [JDXJ(**item) for item in data]
    def get_hscp_cwzb(self, code:str)->list[FinancialReport]:
        """获取公司近一年各季度主要财务指标"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/cwzb/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [FinancialReport(**item) for item in data]
        #return list(map(lambda item:FinancialReport(**item),data))
    def get_hscp_sdgd(self, code:str):
        """获取公司十大股东"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/sdgd/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_ltgd(self, code:str):
        """获取公司十大流通股股东"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/ltgd/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_gdbh(self, code:str):
        """获取公司股东变化趋势"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/gdbh/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hscp_jjcg(self, code:str):
        """获取公司最近500家左右的基金持股情况"""
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/jjcg/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_zlzj(self,code:str):
        """获取某个股票的每分钟主力资金走势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/zlzh/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_zjlr(self,code:str):
        """获取某个股票的近十年每天资金流入趋势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/zjlr/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_zhlrt(self,code:str):
        """获取某个股票的近10天资金流入趋势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/zhlrt/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_jddx(self,code:str):
        """获取某个股票的近十年主力阶段资金动向"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/jddx/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_jddxt(self,code:str):
        """获取某个股票的近十天主力阶段资金动向"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/jddxt/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_lscj(self,code:str):
        """获取某个股票的近十年每天历史成交分布"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/lscj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_lscjt(self,code:str):
        """获取某个股票的近十天历史成交分布"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/lscjt/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_ssjy(self,code:str)-> SSJY:
        """获取某个股票的实时交易数据"""
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/ssjy/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return [SSJY(**item) for item in data]
    def get_hsrl_mmwp(self,code:str,config: RunnableConfig,state: Annotated[dict, InjectedState("mmwp")])->dict:
        """获取某个股票的盘口交易数据,返回值没有当前股价，仅有5档买卖需求量价以及委托统计"""
        #数据更新：交易时间段每2分钟
        #请求频率：1分钟300次
        try:
            code = self._get_stock_code(code)
            if state:
                print("!!!!!-->state.mmwp",state["mmwp"])
                if state["mmwp"]:
                    return state["mmwp"]
            res = requests.get( 
                f"{self.mairui_api_url}/hsrl/mmwp/{code}/{self.mairui_token}",
            )
            data = res.json()
        except Exception as e:
            print("get_hsrl_mmwp?",code)
        
        return data
    def get_hsrl_zbjy(self,code:str):
        """获取某个股票的当天逐笔交易数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/zbjy/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_fscj(self,code:str):
        """获取某个股票的当天分时成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/fscj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_fjcj(self,code:str):
        """获取某个股票的当天分价成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/fjcj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsrl_zbdd(self,code:str):
        """获取某个股票的当天逐笔超400手的大单成交数据"""
        #数据更新：每天20:00开始更新，当日23:59前完成
        #请求频率：1分钟300次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hsrl/zbdd/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hszb_fsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hszb/fsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hszb_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易的平均移动线数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hszb/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_hszbl_fsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hszbl/fsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hszbl_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hszbl/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_hszbc_fsjy(self,code:str,sdt:str,edt:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码某段时间分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hszbc/fsjy/{code}/{fs}/{sdt}/{edt}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hszbc_ma(self,code:str,sdt:str,edt:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码某段时间分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/hszbc/ma/{code}/{fs}/{sdt}/{edt}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_zs_sssj(self,code:str):
        """获取某个指数的实时交易数据"""
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/zs/sssj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_zs_lsgl(self):
        """获取沪深两市不同涨跌幅的股票数统计"""
        #数据更新：交易时间段每2分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        res = requests.get( 
            f"{self.mairui_api_url}/zs/lsgl/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_zs_fsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/zs/fsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_zs_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易的平均移动线数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/zs/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_zs_hfsjy(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/zs/hfsjy/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_zs_hma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取指数代码分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code = self._get_stock_code(code)
        res = requests.get( 
            f"{self.mairui_api_url}/zs/hma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data

    def get_hitc_jrts(self):
        """获取今日股票、基金公告事项以及交易异动概览"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hitc/jrts/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hitc_dzjy(self):
        """获取上一个交易日的大宗交易数据"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hitc/dzjy/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_higg_jlr(self)->JLR:
        """获取所有股票的资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/jlr/{self.mairui_token}",
        )
        data = res.json()        
        return [JLR(**item) for item in data]
    def get_higg_zljlr(self)->ZLJLR:
        """获取所有股票的主力资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/zljlr/{self.mairui_token}",
        )
        data = res.json()        
        return [ZLJLR(**item) for item in data]
    def get_higg_shjlr(self)->SHJLR:
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

    def _ball_code(self,code:str):
        if not code.isnumeric():
            code = self._get_stock_code(code)
        if code.startswith("6"):
            code = "SH"+code
        elif code.startswith("3") or code.startswith("0"):
            code = "SZ" + code
        elif code.startswith("8") or code.startswith("4"):
            code = "BJ" + code
        print("???",code)
        return code

    def quotec(self,code:str):
        '''
        查看股票的实时行情
        '''
        code = self._ball_code(code)
        return ball.quotec(code)
    def pankou(self,code:str):
        '''
        查看股票的实时分笔数据，可以实时取得股票当前报价和成交信息
        '''
        code = self._ball_code(code)
        return ball.pankou(code)
    def capital_flow(self,code:str):
        '''
        获取当日资金流入流出数据，每分钟数据
        '''
        code = self._ball_code(code)
        return ball.capital_flow(code)
    def capital_history(self,code:str):
        '''
        获取历史资金流入流出数据，每日数据
        输出中sum3、sum5、sum10、sum20分别代表3天、5天、10天、20天的资金流动情况
        '''
        return ball.capital_history(code)
    def earningforecast(self,code:str):
        '''
        按年度获取业绩预告数据
        '''
        return ball.earningforecast(code)
    def capital_assort(self,code:str):
        '''
        获取资金成交分布数据
        '''
        code = self._ball_code(code)
        return ball.capital_assort(code)
    def blocktrans(self,code:str):
        '''
        获取大宗交易数据
        '''
        code = self._ball_code(code)
        return ball.blocktrans(code)
    def indicator(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        按年度、季度获取业绩报表数据。
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code = self._ball_code(code)
        return ball.indicator(symbol=code,is_annals=is_annals,count=count)
    def business(self,code:str,*,count:int=10):
        '''
        获取主营业务构成数据
        '''
        code = self._ball_code(code)
        return ball.business(symbol=code,count=count)
    def top_holders(self,code:str,*,circula=1):
        '''
        获取十大股东
        code -> 股票代码
        circula -> 只获取流通股,默认为1
        '''
        code = self._ball_code(code)
        return ball.top_holders(symbol=code,circula=circula)
    def main_indicator(self,code:str):
        '''
        获取主要指标
        '''
        code = self._ball_code(code)
        return ball.main_indicator(code)
    def holders(self,code:str):
        '''
        获取股东人数
        '''
        code = self._ball_code(code)
        return ball.holders(code)
    def org_holding_change(self,code:str):
        '''
        获取机构持仓情况
        '''
        code = self._ball_code(code)
        return ball.org_holding_change(code)
    def industry_compare(self,code:str):
        '''
        获取行业对比数据
        '''
        code = self._ball_code(code)
        return ball.industry_compare(code)
    def income(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取利润表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code = self._ball_code(code)
        return ball.income(symbol=code,is_annals=is_annals,count=count)
    def balance(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取资产负债表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，如果没有指定，可以设定为10条
        '''
        code = self._ball_code(code)
        return ball.balance(symbol=code,is_annals=is_annals,count=count)
    def cash_flow(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取现金流量表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code = self._ball_code(code)
        return ball.cash_flow(symbol=code,is_annals=is_annals,count=count)
    

if __name__ == "__main__":
    from ylz_utils.langchain import LangchainLib
    from ylz_utils.langchain.graph.stock_graph import StockGraph
    
    Config.init('ylz_utils')
    langchainLib = LangchainLib()
    stockGraph = StockGraph(langchainLib)
    # toolLib = SnowballTools(stockGraph)
    # data  = toolLib.balance("ST易联众")
    toolLib = MairuiTools(stockGraph)
    #data = toolLib.get_company_info("ST易联众")
    data = toolLib.get_hsrl_mmwp("瑞芯微")
    print(data)
    if isinstance(data,list):
        print(len(data))
