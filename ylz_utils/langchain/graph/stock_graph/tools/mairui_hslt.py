from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HSLT(MairuiTools):
    def get_hslt_list(self)->list[HSLT_LIST]:
        """获取沪深两市的公司列表"""
        res = requests.get( 
            f"{self.mairui_api_url}/hslt/list/{self.mairui_token}",
        )
        data = res.json()        
        today = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        return [HSLT_LIST(**{**item,"t":today,"dm":f"{item['jys']}{item['dm']}"}) for item in data]
