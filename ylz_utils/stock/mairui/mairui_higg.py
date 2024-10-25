import requests
from . import MairuiStock

class HSMY(MairuiStock):
    def get_higg_jlr(self):
        """获取所有股票的资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/jlr/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_higg_zljlr(self):
        """获取所有股票的主力资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/zljlr/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_higg_shjlr(self):
        """获取所有股票的散户资金净流入"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/higg/shjlr/{self.mairui_token}",
        )
        data = res.json()        
        return data
