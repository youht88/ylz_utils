import requests
from . import MairuiStock

class HIBK(MairuiStock):
    def get_hibk_zjhhy(self):
        """获取所有证监会定义的行业板块个股统计数据"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hibk/zjhhy/{self.mairui_token}",
        )
        data = res.json()      
        return data
    def get_hibk_gnbk(self):
        """获取所有概念板块个股统计数据"""
        #数据更新：每天15:30（约10分钟更新完成）
        #请求频率：1分钟20次
        res = requests.get( 
            f"{self.mairui_api_url}/hibk/gnbk/{self.mairui_token}",
        )
        data = res.json()        
        return data
