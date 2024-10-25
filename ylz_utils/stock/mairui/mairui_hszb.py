from datetime import datetime, timedelta
from typing import Literal
import requests
from . import MairuiStock

class HSZB(MairuiStock):
    def get_hszb_fsjy(self,code:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m",sync_es:bool=False):
        """获取股票代码分时交易实时数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
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
        date_fields = ['ud']
        skip_condition = f"mr_code == '{mr_code}' & (ud.dt.strftime('%Y-%m-%d')>='{ud}')"

        name = f"hszb_fsjy_{mr_code}_{fsjb}"
        df = self._load_data(name,f"hszb/fsjy/{code}/{fsjb}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code"],date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','ud'])
            print(f"errors:{es_result["errors"]}")
        return df


    def get_hszb_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码分时交易的平均移动线数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszb/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_hszbl_fsjy(self,code:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m",sync_es:bool=False):
        """获取股票代码分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
    
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>16:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"mr_code":mr_code,"fsjb":fsjb,"ud":ud}
        date_fields = ['ud','d']
        skip_condition = f"mr_code == '{mr_code}' & fsjb == '{fsjb}' & (ud.dt.strftime('%Y-%m-%d')>='{ud}')"

        name = f"hszbl_fsjy_{fsjb}_{mr_code}"
        df = self._load_data(name,f"hszbl/fsjy/{code}/{fsjb}",
                                        add_fields=add_fields,
                                        skip_condition=skip_condition,
                                        keys=["mr_code","fsjb","d"],date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['mr_code','d'])
            print(f"errors:{es_result["errors"]}")
        return df
    
    def get_hszbl_ma(self,code:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="15m"):
        """获取股票代码分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszbl/ma/{code}/{fs}/{self.mairui_token}",
        )
        data = res.json()        
        return data
    
    def get_hszbc_fsjy(self,code:str,sdt:str,edt:str,fsjb:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="15m"):
        """获取股票代码某段时间分时交易历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
    
        #  http://api.mairui.club/hszbc/fsjy/股票代码(如000001)/分时级别/起始时间/结束时间/您的licence
        if not hasattr(self,"df_hszbc_fsjy"):
            self.df_hszbc_fsjy = pd.DataFrame(columns=HSZBC_FSJY.model_fields.keys())
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        add_fields = {"mr_code":mr_code,"fsjb":fsjb}
        skip_condition = f"mr_code == '{mr_code}' & fsjb == '{fsjb}'"
        df = self._load_data("hszbc_fsjy.csv",f"hszbc/fsjy/{code}/{fsjb}/{sdt}/{edt}",
                                     dataframe=self.df_hszbc_fsjy,
                                     add_fields=add_fields,
                                     skip_condition=skip_condition,
                                     keys=["mr_code",'fsjb'])
        return df
    
    def get_hszbc_ma(self,code:str,sdt:str,edt:str,fs:Literal["5m","15m","30m","60m","dn","wn","mn","yn"]="5m"):
        """获取股票代码某段时间分时交易的平均移动线历史数据。分时级别支持5分钟、15分钟、30分钟、60分钟、日周月年级别，对应的值分别是 5m、15m、30m、60m、dn、wn、mn、yn """
        #数据更新：交易时间段每1分钟
        #请求频率：1分钟600次 | 包年版1分钟3千次 | 钻石版1分钟6千次
        code_info = self._get_stock_code(code)
        code=code_info['code']
        mr_code = code_info['mr_code']
        res = requests.get( 
            f"{self.mairui_api_url}/hszbc/ma/{code}/{fs}/{sdt}/{edt}/{self.mairui_token}",
        )
        data = res.json()        
        return data

