from . import GraphLib
import pysnowball as ball

from ylz_utils.config import Config 

class Tools:
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib
        token = Config.get('STOCK.SNOWBALL.TOKEN')
        print(ball.set_token(f"xq_a_token={token};"))
    def quotec(self,code:str):
        '''
        查看股票的实时行情
        '''
        return ball.quotec(code)
    def pankou(self,code:str):
        '''
        查看股票的实时分笔数据，可以实时取得股票当前报价和成交信息
        '''
        return ball.pankou(code)
    def capital_flow(self,code:str):
        '''
        获取当日资金流入流出数据，每分钟数据
        '''
        return ball.capital_flow(code)
    def capital_assort(self,code:str):
        '''
        获取资金成交分布数据
        '''
        return ball.capital_assort(code)
    def blocktrans(self,code:str):
        '''
        获取大宗交易数据
        '''
        return ball.blocktrans(code)
    def indicator(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        按年度、季度获取业绩报表数据。
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        return ball.indicator(symbol=code,is_annals=is_annals,count=count)
    def business(self,code:str,*,count:int=10):
        '''
        获取主营业务构成数据
        '''
        return ball.business(symbol=code,count=count)
    def top_holders(self,code:str,*,circula=1):
        '''
        获取十大股东
        code -> 股票代码
        circula -> 只获取流通股,默认为1
        '''
        return ball.top_holders(symbol=code,circula=circula)
    def main_indicator(self,code:str):
        '''
        获取主要指标
        '''
        return ball.main_indicator(code)
    def holders(self,code:str):
        '''
        获取股东人数
        '''
        return ball.holders(code)
    def org_holding_change(self,code:str):
        '''
        获取机构持仓情况
        '''
        return ball.org_holding_change(code)
    def industry_compare(seld,code:str):
        '''
        获取行业对比数据
        '''
        return ball.industry_compare(code)
    def income(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取利润表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        return ball.income(symbol=code,is_annals=is_annals,count=count)
    def balance(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取资产负债表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，如果没有指定，可以设定为10条
        '''
        return ball.balance(symbol=code,is_annals=is_annals,count=count)
    def cash_flow(self,code:str,*,is_annals:int=1,count:int=10):
        '''
        获取现金流量表
        code -> 股票代码
        is_annals -> 只获取年报,默认为1
        count -> 返回数据数量，默认10条
        '''
        return ball.cash_flow(symbol=code,is_annals=is_annals,count=count)
    