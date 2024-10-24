from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HSRL(MairuiTools):
    def get_hsrl_ssjy(self,code:str):
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
        #return [SSJY(**item) for item in data]
    def get_hsrl_mmwp(self,code:str,sync_es:bool=False):
        """获取某个股票的盘口交易数据,返回值没有当前股价，仅有5档买卖需求量价以及委托统计"""
        #数据更新：交易时间段每2分钟
        #请求频率：1分钟300次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        date_fields = ['t']
        skip_condition = None
        name = f"hsrl_mmwp_{mr_code}"
        df = self._load_data(name,f"hsrl/mmwp/{code}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code","t"],date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','t'])
            print(f"errors:{es_result["errors"]}")
        return df

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

if __name__ == "__main__":
    from ylz_utils.langchain import LangchainLib
    from ylz_utils.langchain.graph.stock_graph import StockGraph
    import time

    Config.init('ylz_utils')
    langchainLib = LangchainLib()
    stockGraph = StockGraph(langchainLib)

    lib = HSRL(stockGraph)
    for i in range(5):
        res = lib._parallel_execute(lib.get_hsrl_mmwp,['全志科技','瑞芯微','欧菲光','宗申动力','银邦股份','蒙草生态','万达信息'],sync_es=True)
        print(f"****** {i} , {len(res)}  ******")
        time.sleep(120)