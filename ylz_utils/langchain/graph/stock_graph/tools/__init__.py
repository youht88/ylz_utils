from datetime import datetime,timedelta
from typing import Annotated, Literal
import pysnowball as ball
import tushare
import pandas as pd
import requests
import json
import os
from ylz_utils.config import Config 

from rich import print
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from ylz_utils.database.elasticsearch import ESLib
from ylz_utils.langchain.graph.stock_graph.state import *

import concurrent.futures
import time


class StockTools:
    stock:list = []
    def __init__(self,graphLib):
        self.graphLib = graphLib
        self.esLib = ESLib(using='es')
        # 获取当前模块的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建 gpdm 文件的完整路径
        gpdm_file = os.path.join(current_dir, 'gpdm.json')
        with open(gpdm_file, 'r', encoding='utf-8') as f:
            self.gpdm = json.load(f)
         # 构建 zsdm 文件的完整路径
        zsdm_file = os.path.join(current_dir, 'zsdm.json')
        with open(zsdm_file, 'r', encoding='utf-8') as f:
            self.zsdm = json.load(f)
    # 定义一个执行函数的方法
    def _parallel_execute(self,fun:callable,codes,**kwargs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fun,code,**kwargs) for code in codes]  # 并发执行10次函数
            results = [future.result() for future in futures]
            return results

    def _get_stock_code(self,stock_name:str)->dict:
        """根据股票或指数名称获取股票/指数代码"""
        if stock_name.startswith('sh') or stock_name.startswith('sz'):
            # mairui code
            jys = stock_name[:2]
            code = stock_name[2:]
            ts_code = f"{code}.{jys.upper()}"
            mr_code = stock_name
            ball_code = f"{jys.upper()}{code}"
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                name = stock_info[0]['name']
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        elif stock_name.endswith('.SH') or stock_name.endswith('.SZ'):
            #tu-share code
            jys = stock_name[-2:].lower()
            code = stock_name[:-3]
            ts_code = stock_name
            mr_code = f"{jys}{code}"
            ball_code = f"{jys.upper()}{code}"
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                name = stock_info[0]['name']
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        elif stock_name.startswith('SH') or stock_name.startswith('SZ'):
            #雪球code
            jys = stock_name[:2].lower()
            code = stock_name[2:]
            ball_code = stock_name
            ts_code = f"{code}.{jys.upper()}"
            mr_code = f"{jys}{code}"
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                name = stock_info[0]['name']
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        elif stock_name.isnumeric():
            code = stock_name
            stock_info = list(filter(lambda item:item["symbol"]==code,self.gpdm))
            if stock_info:
                ts_code = stock_info[0]['ts_code']
                jys = ts_code[-2:].lower()
                mr_code = f"{jys}{code}"
                name = stock_info[0]['name']
                ball_code = f"{jys.upper()}{code}"
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                return {}
        else:
            stock_info = list(filter(lambda item:item["name"]==stock_name.upper(),self.gpdm))
            if stock_info:
                ts_code = stock_info[0]['ts_code']
                jys = ts_code[-2:].lower()
                code = stock_info[0]['symbol']
                name = stock_info[0]['name']
                mr_code = f"{jys}{code}"
                ball_code = f"{jys.upper()}{code}"
                return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
            else:
                zs_info = list(filter(lambda item:item["mc"]==stock_name.upper(),self.zsdm))
                if zs_info:
                    mr_code = zs_info[0]['dm']
                    jys = zs_info[0]['jys']
                    name = zs_info[0]['mc']
                    code = mr_code[2:]
                    ts_code = f"{code}.{jys.upper()}"
                    ball_code = f"{jys.upper()}{code}"
                    return {"code":code,"mr_code":mr_code,"ts_code":ts_code,"name":name,"jys":jys,"ball_code":ball_code}
 
        
class MairuiTools(StockTools):
    def __init__(self,graphLib):
        super().__init__(graphLib)
        self.mairui_token = Config.get('STOCK.MAIRUI.TOKEN')
        print("MAIRUI_TOKEN=",self.mairui_token)
        self.mairui_api_url = "http://api.mairui.club" 
        self._exports = [
            'get_hscp_cwzb','get_hscp_jdlr','get_hscp_jdxj',
            'get_hsmy_jddxt','get_hsmy_lscjt',
            'get_hsrl_zbdd','get_hsrl_mmwp',
            'get_hscp_.*',
            '.*'
        ] 
        self.dataframe_map:dict[str,pd.DataFrame]={}       

    def _load_data(self,name:str,method_path:str,
                           add_fields:dict={},skip_condition:str=None,keys=[],date_fields=[])->pd.DataFrame:
        # dataframe 传入的类实例变量，用于多次检索时的内存cache
        # add_fields 需要增加的字段，例如 add_fields = {"mr_code":"abcd"}
        # skip_condition 过滤的字符串，如果过滤条件下没有数据则需要从network中获取
        # keys 判断是否重复数据的字段名数组,例如 keys=["mr_code"]
        # date_fields 指定日期类型的字段名
        try:
            file_name = f"{name}.csv"
            df_name = f"df_{name}"             
            if not isinstance(self.dataframe_map.get(df_name),pd.DataFrame):
                self.dataframe_map[df_name] = pd.DataFrame([])
            dataframe = self.dataframe_map[df_name]
            df=None
            get_df = None
            cache_df = None
            if dataframe.empty:
                try:
                    dataframe = pd.read_csv(file_name,parse_dates=date_fields)
                    for col in date_fields:
                        dataframe[col] = pd.to_datetime(dataframe[col])
                    # 判断是否rload
                    print("skip_condition:",skip_condition)
                    if not skip_condition:
                        print("!!!!! ALWAYS LOAD FROM NETWORK !!!!")
                        raise Exception("always reload")
                    cache_df = dataframe.query(skip_condition) 
                    if cache_df.empty:
                        print("!!!!! NEED LOAD FROM NETWORK !!!!!")
                        raise Exception("need load from network")
                except Exception as e:
                    print("start retrieve data from network...",e)
                    try:
                        res = requests.get(f"{self.mairui_api_url}/{method_path}/{self.mairui_token}")
                        data = res.json()
                        if isinstance(data,list):
                            data = [{**item,**add_fields} for item in data]
                        else:
                            data = [{**data,**add_fields}]
                        get_df = pd.DataFrame(data)
                        for col in date_fields:
                            get_df[col] = pd.to_datetime(get_df[col]) 
                        if dataframe.empty:
                            dataframe = get_df
                            dataframe.to_csv(file_name,index=False)                       
                    except Exception as e:
                        print("network error:",e)
                        raise  
            if not (get_df is None):
                print("get_df.count=",get_df.shape)                  
            if not (cache_df is None):
                print("cache_df.count=",cache_df.shape)
            if not (dataframe is None):
                print("dataframe.count=",dataframe.shape)                  
            if not dataframe.empty:
                if not (get_df is None):
                    condition = pd.Series([True] * len(dataframe))
                    # 根据字段名列表构建动态筛选条件
                    for k in keys:
                        if k in get_df.columns:
                            condition = condition & (dataframe[k] == get_df[k][0])
                    # 应用筛选条件
                    find_row:pd.DataFrame = dataframe[condition]
                    if find_row.empty:
                        dataframe = pd.concat([dataframe,get_df], ignore_index=True)
                        dataframe.to_csv(file_name,index=False)
                    df = get_df
                elif not (cache_df is None):
                    df = cache_df
        except Exception as e:
            raise Exception(f"error on _load_data_by_code,the error is :{e}")
        return df


class TushareTools(StockTools):
    def __init__(self,graphLib):
        super().__init__(graphLib)
        self.tushare_token = Config.get('STOCK.TUSHARE.TOKEN')
        self.tuapi = tushare.pro_api(self.tushare_token)
        self.tushare_api_url = "http://api.tushare.pro"

    def _tushare_query(self, api_name, fields='', **kwargs):
        req_params = {
            'api_name': api_name,
            'token': self.tushare_token,
            'params': kwargs,
            'fields': fields
        }
        res = requests.post(
            self.tushare_api_url,
            req_params
        )

        result = json.loads(res.read().decode('utf-8'))

        if result['code'] != 0:
            raise Exception(result['msg'])

        data  = result['data']
        columns = data['fields']
        items = data['items']

        return pd.DataFrame(items, columns=columns)

class SnowballTools(StockTools): 
    def __init__(self,graphLib):
        super().__init__(graphLib)
        snowball_token = Config.get('STOCK.SNOWBALL.TOKEN')
        ball.set_token(f"xq_a_token={snowball_token};") 
        print(f"snowball token:{snowball_token}")

    def quotec(self,code:str):
        '''
        查看股票的实时行情
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.quotec(ball_code)
    def pankou(self,code:str):
        '''
        查看股票的实时分笔数据，可以实时取得股票当前报价和成交信息
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        res =  ball.pankou(ball_code)
        data ={ 
        "t":datetime.fromtimestamp(res['data']['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S'),
        "mr_code":mr_code,
        "vc":res['data']['diff']/100,
        "vb":res['data']['ratio'],
        "pb1":res['data']['bp1'],
        "vb1":res['data']['bc1']/100,
        "pb2":res['data']['bp2'],
        "vb2":res['data']['bc2']/100,
        "pb3":res['data']['bp3'],
        "vb3":res['data']['bc3']/100,
        "pb4":res['data']['bp4'],
        "vb4":res['data']['bc4']/100,
        "pb5":res['data']['bp5'],
        "vb5":res['data']['bc5']/100,
        "ps1":res['data']['sp1'],
        "vs1":res['data']['sc1']/100,
        "ps2":res['data']['sp2'],
        "vs2":res['data']['sc2']/100,
        "ps3":res['data']['sp3'],
        "vs3":res['data']['sc3']/100,
        "ps4":res['data']['sp4'],
        "vs4":res['data']['sc4']/100,
        "ps5":res['data']['sp5'],
        "vs5":res['data']['sc5']/100,
        }
        #return [HSRL_MMWP(**item) for item in [data]]
        return data
    def capital_flow(self,code:str):
        '''
        获取当日资金流入流出数据，每分钟数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return [ball.capital_flow(ball_code)]
    def capital_history(self,code:str):
        '''
        获取历史资金流入流出数据，每日数据
        输出中sum3、sum5、sum10、sum20分别代表3天、5天、10天、20天的资金流动情况
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.capital_history(ball_code)
    def earningforecast(self,code:str):
        '''
        按年度获取业绩预告数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.earningforecast(ball_code)
    def capital_assort(self,code:str):
        '''
        获取资金成交分布数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.capital_assort(ball_code)
    def blocktrans(self,code:str):
        '''
        获取大宗交易数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.blocktrans(ball_code)
    def indicator(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        按年度、季度获取业绩报表数据。
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.indicator(symbol=ball_code,is_annals=is_annals,count=count)
    def business(self,code:str,*,count:int=10):
        '''
        获取主营业务构成数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.business(symbol=ball_code,count=count)
    def top_holders(self,code:str,*,circula=1):
        '''
        获取十大股东
        code -> 股票代码
        circula -> 只获取流通股,默认为1
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.top_holders(symbol=ball_code,circula=circula)
    def main_indicator(self,code:str):
        '''
        获取主要指标
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.main_indicator(ball_code)
    def holders(self,code:str):
        '''
        获取股东人数
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.holders(ball_code)
    def org_holding_change(self,code:str):
        '''
        获取机构持仓情况
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.org_holding_change(ball_code)
    def industry_compare(self,code:str):
        '''
        获取行业对比数据
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.industry_compare(ball_code)
    def income(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取利润表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.income(symbol=ball_code,is_annals=is_annals,count=count)
    def balance(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取资产负债表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，如果没有指定，可以设定为10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.balance(symbol=ball_code,is_annals=is_annals,count=count)
    def cash_flow(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取现金流量表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        return ball.cash_flow(symbol=ball_code,is_annals=is_annals,count=count)

if __name__ == "__main__":
    from ylz_utils.langchain import LangchainLib
    from ylz_utils.langchain.graph.stock_graph import StockGraph
    import time

    Config.init('ylz_utils')
    langchainLib = LangchainLib()
    stockGraph = StockGraph(langchainLib)
    print("雪球--->")
    lib = SnowballTools(stockGraph)
    
    for i in range(10):
        res1 = lib._parallel_execute(lib.pankou,['全志科技','欧菲光'])
        print(f"****** {i} , {len(res1)}  ******")
        print(res1[0]['t'],res1[0]['mr_code'],res1[0]['vc'],res1[0]['vb'])
        #lib.esLib.save("capital_flow",[res1],ids=['mr_code','t'])
        time.sleep(5)