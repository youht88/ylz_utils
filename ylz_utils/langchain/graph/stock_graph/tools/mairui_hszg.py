from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HSZG(MairuiTools):
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
