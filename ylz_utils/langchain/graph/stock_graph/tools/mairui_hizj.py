from datetime import datetime, timedelta
import requests
from rich import print
from ylz_utils.config import Config
from ylz_utils.langchain.graph.stock_graph.tools import MairuiTools

class HIZJ(MairuiTools):
    def get_hizj_zjh(self,sync_es:bool=False):   
        """获取近3、5、10天证监会行业资金流入情况"""
        #数据更新：每天15:30开始更新（更新耗时约10分钟）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"hizj_zjh"
        
        df = self._load_data(name,f"hizj/zjh",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df
    def get_hizj_bk(self,sync_es:bool=False):   
        """获取近3、5、10天概念板块资金流入情况"""
        #数据更新：每天15:30开始更新（更新耗时约10分钟）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"hizj_bk"
        
        df = self._load_data(name,f"hizj/bk",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df
    
    def get_hizj_ggzl(self,sync_es:bool=False):   
        """个股阶段净流入资金统计总览"""
        #数据更新：每天15:30开始更新（更新耗时约10分钟）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"hizj_ggzl"
        
        df = self._load_data(name,f"hizj/ggzl",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df
    
    def get_hizj_lxlr(self,sync_es:bool=False):   
        """获取主力连续净流入统计"""
        #数据更新：每天15:30开始更新（更新耗时约10分钟）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"hizj_lxlr"
        
        df = self._load_data(name,f"hizj/lxlr",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df

    def get_hizj_lxlc(self,sync_es:bool=False):   
        """获取主力连续净流出统计"""
        #数据更新：每天15:30开始更新（更新耗时约10分钟）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"hizj_lxlc"
        
        df = self._load_data(name,f"hizj/lxlc",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df

if __name__ == "__main__":
    from ylz_utils.langchain import LangchainLib
    from ylz_utils.langchain.graph.stock_graph import StockGraph
    import time

    Config.init('ylz_utils')
    langchainLib = LangchainLib()
    stockGraph = StockGraph(langchainLib)
    toolLib = HIZJ(stockGraph)

    data = toolLib.get_hizj_zjh(sync_es=True)
    print("证监会板块资金流入情况\n",data.head(20))
    if isinstance(data,list):
        print(len(data))
    
    data = toolLib.get_hizj_bk(sync_es=True)
    print("概念资金流入情况\n",data.head(20))
    if isinstance(data,list):
        print(len(data))
    
    data = toolLib.get_hizj_ggzl(sync_es=True)
    print("个股资金流入情况总览入\n",data.head(20))
    if isinstance(data,list):
        print(len(data))

    data = toolLib.get_hizj_lxlr(sync_es=True)
    print("主力连续流入\n",data.head(20))
    if isinstance(data,list):
        print(len(data))

    data = toolLib.get_hizj_lxlc(sync_es=True)
    print("主力连续流出\n",data.head(20))
    if isinstance(data,list):
        print(len(data))

    #result = toolLib.esLib.sql("select avg(o),avg(c) from hszbl_fsjy_dn_sh603893 where d>'2024-10-20'")
    #print(result)
    
