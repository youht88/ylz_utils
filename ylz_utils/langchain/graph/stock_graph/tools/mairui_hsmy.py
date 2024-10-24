from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HSMY(MairuiTools):
    def get_hsmy_zlzj(self,code:str):
        """获取某个股票的每分钟主力资金走势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        skip_condition = f"mr_code == '{mr_code}'"
        date_fields = ['t']
        df = self._load_data("hsmy_zlzj",f"hsmy/zlzj/{code}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code"],date_fields=date_fields)
        return df
    def get_hsmy_zjlr(self,code:str):
        """获取某个股票的近十年每天资金流入趋势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code}
        skip_condition = f"mr_code == '{mr_code}'"
        date_fields = ['t']
        df = self._load_data("hsmy_zjlr",f"hsmy/zjlr/{code}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code"],date_fields=date_fields)
        return df

    def get_hsmy_zhlrt(self,code:str,sync_es:bool=False):
        """获取某个股票的近10天资金流入趋势"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"mr_code":mr_code,"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"mr_code == '{mr_code}' & (ud.dt.strftime('%Y-%m-%d')>='{ud}')"

        name = f"hsmy_zhlrt_{mr_code}"
        df = self._load_data(name,f"hsmy/zhlrt/{code}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code"],date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','t','ud'])
            print(f"errors:{es_result["errors"]}")
        return df

    def get_hsmy_jddx(self,code:str):
        """获取某个股票的近十年主力阶段资金动向"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/jddx/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_jddxt(self,code:str):
        """获取某个股票的近十天主力阶段资金动向"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/jddxt/{code}/{self.mairui_token}",
        )
        if res.status_code==200:
            data = res.json()        
            return data
        else:
            return []
    def get_hsmy_lscj(self,code:str):
        """获取某个股票的近十年每天历史成交分布"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/lscj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    def get_hsmy_lscjt(self,code:str):
        """获取某个股票的近十天历史成交分布"""
        #数据更新：每天20:00开始更新（更新耗时约4小时）
        #请求频率：1分钟300次 
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hsmy/lscjt/{code}/{self.mairui_token}",
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
    # toolLib = SnowballTools(stockGraph)
    # data  = toolLib.balance("ST易联众")
    toolLib = HSMY(stockGraph)

    #data1=data2=data3=data4=data5=[]
    # for i in range(30):
    #     print("index======>",i,len(data1),len(data2),len(data3),len(data4),len(data5))
    #     data1 = toolLib.get_hsrl_mmwp("中粮资本")
    #     data2 = toolLib.get_hsrl_mmwp("蒙草生态")
    #     data3 = toolLib.get_hsrl_mmwp("瑞芯微")
    #     data4 = toolLib.get_hsrl_mmwp("万达信息")
    #     data5 = toolLib.get_hsrl_mmwp("全志科技")
    #     time.sleep(60)
    #data = toolLib.get_hsrl_zbjy("万达信息")
    #data = toolLib.get_hsrl_zbjy("万达信息")
    #data = toolLib.get_zs_hfsjy("蒙草生态")
    #data = toolLib.get_zs_lsgl()
    #data = toolLib.get_hszg_zg("旗天科技")
    #data = toolLib.get_hscp_cwzb("蒙草生态")
    #data = toolLib.get_hsrl_mmwp("福日电子")
    #data = toolLib.get_hszbc_fsjy("蒙草生态","2024-08-30","2024-08-30",'5m')
    
    # data = toolLib.get_hsmy_zlzj("蒙草生态")
    # print(data.head(5))
    # if isinstance(data,list):
    #     print(len(data))

    # data = toolLib.get_hsmy_zjlr("蒙草生态")
    # print(data.head(5))
    # if isinstance(data,list):
    #     print(len(data))

    data = toolLib.get_hsmy_zhlrt("瑞芯微",sync_es=True)
    print(data.head(20))
    if isinstance(data,list):
        print(len(data))

    result = toolLib.esLib.count("hsmy_zhlrt_sz001696",{"query":{"match_all":{}}})
    print(result)

    


