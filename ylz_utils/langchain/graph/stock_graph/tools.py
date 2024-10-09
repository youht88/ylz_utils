import pysnowball as ball
import tushare
import pandas as pd
import requests
import json
import os
from ylz_utils.config import Config 

from pydantic import BaseModel, Field
from datetime import datetime
from rich import print

class CompanyInfo(BaseModel):
    name: str = Field(..., description="公司名称")
    ename: str = Field(..., description="公司英文名称")
    market: str = Field(..., description="上市市场")
    idea: str = Field(..., description="概念及板块，多个概念由英文逗号分隔")
    ldate: datetime = Field(..., description="上市日期，格式yyyy-MM-dd")
    sprice: str = Field(..., description="发行价格（元）")
    principal: str = Field(..., description="主承销商")
    rdate: str = Field(..., description="成立日期")
    rprice: str = Field(..., description="注册资本")
    instype: str = Field(..., description="机构类型")
    organ: str = Field(..., description="组织形式")
    secre: str = Field(..., description="董事会秘书")
    phone: str = Field(..., description="公司电话")
    sphone: str = Field(..., description="董秘电话")
    fax: str = Field(..., description="公司传真")
    sfax: str = Field(..., description="董秘传真")
    email: str = Field(..., description="公司电子邮箱")
    semail: str = Field(..., description="董秘电子邮箱")
    site: str = Field(..., description="公司网站")
    post: str = Field(..., description="邮政编码")
    infosite: str = Field(..., description="信息披露网址")
    oname: str = Field(..., description="证券简称更名历史")
    addr: str = Field(..., description="注册地址")
    oaddr: str = Field(..., description="办公地址")
    desc: str = Field(..., description="公司简介")
    bscope: str = Field(..., description="经营范围")
    printype: str = Field(..., description="承销方式")
    referrer: str = Field(..., description="上市推荐人")
    putype: str = Field(..., description="发行方式")
    pe: str = Field(..., description="发行市盈率（按发行后总股本）")
    firgu: str = Field(..., description="首发前总股本（万股）")
    lastgu: str = Field(..., description="首发后总股本（万股）")
    realgu: str = Field(..., description="实际发行量（万股）")
    planm: str = Field(..., description="预计募集资金（万元）")
    realm: str = Field(..., description="实际募集资金合计（万元）")
    pubfee: str = Field(..., description="发行费用总额（万元）")
    collect: str = Field(..., description="募集资金净额（万元）")
    signfee: str = Field(..., description="承销费用（万元）")
    pdate: datetime = Field(..., description="招股公告日")

class StockTools:
    stock:list = []
    def __init__(self,graphLib):
        self.graphLib = graphLib
        
        # 获取当前模块的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建 JSON 文件的完整路径
        stock_file = os.path.join(current_dir, 'stock.json')
        with open(stock_file, 'r', encoding='utf-8') as f:
            self.stock = json.load(f)

class MairuiTools(StockTools):
    def __init__(self,graphLib):
        super().__init__(graphLib)
        self.mairui_token = Config.get('STOCK.MAIRUI.TOKEN')
        self.mairui_api_url = "http://api.mairui.club"
        
    def get_company_info(self, code:str)->CompanyInfo:
        """获取公司基本信息"""
        res = requests.get( 
            f"{self.mairui_api_url}/hscp/gsjj/{code}/{self.mairui_token}",
        )
        data = res.json()        
        return CompanyInfo(**data)
    
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

    def _get_stock_code(self,stock_name):
        """根据股票名称获取股票代码"""
        stock_info = list(filter(lambda item:item["name"]==stock_name,self.stock))
        if stock_info:
            stock_code = stock_info[0]['symbol']
            return stock_code
        else:
            return ""
    def _ball_code(self,code:str):
        if not code.isnumeric():
            code = self._get_stock_code(code)
        if code.startswith("6"):
            code = "SH"+code
        elif code.startswith("3") or code.startswith("0"):
            code = "SZ" + code
        elif code.startswith("8") or code.startswith("4"):
            code = "BJ" + code
        print("???",code)
        return code

    def quotec(self,code:str):
        '''
        查看股票的实时行情
        '''
        code = self._ball_code(code)
        return ball.quotec(code)
    def pankou(self,code:str):
        '''
        查看股票的实时分笔数据，可以实时取得股票当前报价和成交信息
        '''
        code = self._ball_code(code)
        return ball.pankou(code)
    def capital_flow(self,code:str):
        '''
        获取当日资金流入流出数据，每分钟数据
        '''
        code = self._ball_code(code)
        return ball.capital_flow(code)
    def capital_history(self,code:str):
        '''
        获取历史资金流入流出数据，每日数据
        输出中sum3、sum5、sum10、sum20分别代表3天、5天、10天、20天的资金流动情况
        '''
        return ball.capital_history(code)
    def earningforecast(self,code:str):
        '''
        按年度获取业绩预告数据
        '''
        return ball.earningforecast(code)
    def capital_assort(self,code:str):
        '''
        获取资金成交分布数据
        '''
        code = self._ball_code(code)
        return ball.capital_assort(code)
    def blocktrans(self,code:str):
        '''
        获取大宗交易数据
        '''
        code = self._ball_code(code)
        return ball.blocktrans(code)
    def indicator(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        按年度、季度获取业绩报表数据。
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code = self._ball_code(code)
        return ball.indicator(symbol=code,is_annals=is_annals,count=count)
    def business(self,code:str,*,count:int=10):
        '''
        获取主营业务构成数据
        '''
        code = self._ball_code(code)
        return ball.business(symbol=code,count=count)
    def top_holders(self,code:str,*,circula=1):
        '''
        获取十大股东
        code -> 股票代码
        circula -> 只获取流通股,默认为1
        '''
        code = self._ball_code(code)
        return ball.top_holders(symbol=code,circula=circula)
    def main_indicator(self,code:str):
        '''
        获取主要指标
        '''
        code = self._ball_code(code)
        return ball.main_indicator(code)
    def holders(self,code:str):
        '''
        获取股东人数
        '''
        code = self._ball_code(code)
        return ball.holders(code)
    def org_holding_change(self,code:str):
        '''
        获取机构持仓情况
        '''
        code = self._ball_code(code)
        return ball.org_holding_change(code)
    def industry_compare(self,code:str):
        '''
        获取行业对比数据
        '''
        code = self._ball_code(code)
        return ball.industry_compare(code)
    def income(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取利润表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code = self._ball_code(code)
        return ball.income(symbol=code,is_annals=is_annals,count=count)
    def balance(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取资产负债表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，如果没有指定，可以设定为10条
        '''
        code = self._ball_code(code)
        return ball.balance(symbol=code,is_annals=is_annals,count=count)
    def cash_flow(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取现金流量表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        code = self._ball_code(code)
        return ball.cash_flow(symbol=code,is_annals=is_annals,count=count)
    

if __name__ == "__main__":
    from ylz_utils.langchain import LangchainLib
    from ylz_utils.langchain.graph.stock_graph import StockGraph
    
    Config.init('ylz_utils')
    langchainLib = LangchainLib()
    stockGraph = StockGraph(langchainLib)
    # toolLib = SnowballTools(stockGraph)
    # data  = toolLib.balance("ST易联众")
    toolLib = MairuiTools(stockGraph)
    data = toolLib.get_company_info("300096")
    print(data)
