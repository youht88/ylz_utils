from datetime import datetime, timedelta
import requests
from . import MairuiStock

class HIZJ(MairuiStock):
    def get_hizj_zjh(self,sync_es:bool=False):   
        """获取近3、5、10天证监会行业资金流入情况"""
        #数据更新：每天15:30开始更新（更新耗时约10分钟）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        print("run get_hizj_zjh",today)
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
