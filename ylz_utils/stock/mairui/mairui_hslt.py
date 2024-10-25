import requests
from . import MairuiStock
class HSLT(MairuiStock):
    def get_hslt_list(self):
        """获取沪深两市的公司列表"""
        res = requests.get( 
            f"{self.mairui_api_url}/hslt/list/{self.mairui_token}",
        )
        data = res.json()        
        return data

