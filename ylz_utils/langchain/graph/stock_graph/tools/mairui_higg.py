from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HSMY(MairuiTools):
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
