from datetime import datetime, timedelta
import requests
from . import MairuiStock

class HSZG(MairuiStock):
    def get_hszg_list(self):
        """获取沪深两市的指数代码"""
        res = requests.get( 
            f"{self.mairui_api_url}/hszg/list/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hszg_gg(self,code:str):
        '''根据指数、行业、概念板块代码找股票'''
        #http://api.mairui.club/hszg/gg/指数代码/您的licence
        code_info = self._get_bk_code(code)
        code = code_info["code"]
        res = requests.get( 
            f"{self.mairui_api_url}/hszg/gg/{code}/{self.mairui_token}",
        )
        data = res.json() 
        return data
    def get_hszg_zg(self,code:str):
        '''根据股票找相关指数、行业、概念板块'''
        #http://api.mairui.club/hszg/zg/股票代码(如000001)/您的licence
        code_info = self._get_stock_code(code)
        code = code_info["code"]
        res = requests.get( 
            f"{self.mairui_api_url}/hszg/zg/{code}/{self.mairui_token}",
        )
        data = res.json() 
        return data

