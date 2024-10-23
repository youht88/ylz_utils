from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HITC(MairuiTools):
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
