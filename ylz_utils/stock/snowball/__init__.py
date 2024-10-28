from datetime import datetime

from fastapi import APIRouter
import pysnowball as ball

from ylz_utils.config import Config
from ylz_utils.stock import StockLib

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

class SnowballStock(StockLib): 
    def __init__(self):
        super().__init__()
        snowball_token = Config.get('STOCK.SNOWBALL.TOKEN')
        ball.set_token(f"xq_a_token={snowball_token};") 
        print(f"snowball token:{snowball_token}")
    def watch_list(self):
        '''获取用户自选列表'''
        data = ball.watch_list()
        return data
    def watch_stock(self,id):
        '''获取用户自选详情'''
        data = ball.watch_stock(id)
        return data
    def quotec(self,code:str):
        '''
        查看股票的实时行情
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        name = code_info['name']
        res = ball.quotec(ball_code)
        data =  { **res['data']['quote'],
                  'mr_code': mr_code,
                  't':datetime.fromtimestamp(res['data']['quote']['timestamp']/1000),
                  'name': name,
                }
        # data = {
        #     "mr_code": mr_code,
        #     "t":datetime.fromtimestamp(res['data'][0]['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S'),
        #     "name": name,
        #     "current": res['data'][0]['current'],
        #     "percent": res['data'][0]['percent'],
        #     'chg': res['data'][0]['chg'],
        #     'volume': res['data'][0]['volume'],
        #     'amount': res['data'][0]['amount'],
        #     'market_capital': res['data'][0]['market_capital'],
        #     'float_market_capital': res['data'][0]['turnover_rate'],
        #     'turnover_rate': res['data'][0]['turnover_rate'],
        #     'amplitude': res['data'][0]['amplitude'],
        #     'open': res['data'][0]['open'],
        #     'last_close': res['data'][0]['last_close'],
        #     'high': res['data'][0]['high'],
        #     'low': res['data'][0]['low'],
        #     'avg_price': res['data'][0]['avg_price'],
        #     'trade_volume': res['data'][0]['trade_volume'],
        #     'side': res['data'][0]['side'],
        #     'is_trade': res['data'][0]['is_trade'],
        #     'level': res['data'][0]['level'],
        #     'trade_session': res['data'][0]['trade_session'],
        #     'trade_type': res['data'][0]['trade_type'],
        #     'current_year_percent': res['data'][0]['current_year_percent'],
        #     'trade_unique_id': res['data'][0]['trade_unique_id'],
        #     'type': res['data'][0]['type'],
        #     'bid_appl_seq_num': res['data'][0]['bid_appl_seq_num'],
        #     'offer_appl_seq_num': res['data'][0]['offer_appl_seq_num'],
        #     'volume_ext': res['data'][0]['volume_ext'],
        #     'traded_amount_ext': res['data'][0]['traded_amount_ext'],
        #     'trade_type_v2': res['data'][0]['trade_type_v2'],
        #     'yield_to_maturity': res['data'][0]['yield_to_maturity']
        # }
        return data
    def quotec_detail(self,code:str,sync_es:bool=False):
        '''
        查看股票的实时行情细节
        '''
        code_info = self._get_stock_code(code)
        ball_code=code_info['ball_code']
        mr_code = code_info['mr_code']
        name = code_info['name']
        res = ball.quote_detail(ball_code)
        data =  { **res['data']['quote'],
                  'mr_code': mr_code,
                  't':datetime.fromtimestamp(res['data']['quote']['timestamp']/1000),
                  'name': name,
                  'pankou_ratio': res['data']['others'].get('pankou_ratio'),
                  'cyb_switch': res['data']['others'].get('cyb_switch')
                }
        # data = {
        #     "mr_code": mr_code,
        #     "t":datetime.fromtimestamp(res['data']['quote']['timestamp']/1000),
        #     "name": name,
        #     # 与quotec相同
        #     'volume_ext': res['data']['quote']['volume_ext'],
        #     'type': res['data']['quote']['type'],
        #     'high': res['data']['quote']['high'],
        #     'float_market_capital': res['data']['quote']['turnover_rate'],
        #     'chg': res['data']['quote']['chg'],
        #     'last_close': res['data']['quote']['last_close'],
        #     'volume': res['data']['quote']['volume'],
        #     'turnover_rate': res['data']['quote']['turnover_rate'],
        #     'avg_price': res['data']['quote']['avg_price'],
        #    "percent": res['data']['quote']['percent'],
        #     'amplitude': res['data']['quote']['amplitude'],
        #     "current": res['data']['quote']['current'],
        #     'current_year_percent': res['data']['quote']['current_year_percent'],
        #     'low': res['data']['quote']['low'],
        #     'market_capital': res['data']['quote']['market_capital'],
        #     'amount': res['data']['quote']['amount'],
        #     'traded_amount_ext': res['data']['quote']['traded_amount_ext'],
        #     'open': res['data']['quote']['open'],
        #     # 增加的detail
        #     'current_ext': res['data']['quote']['current_ext'],
        #     'high52w': res['data']['quote']['high52w'],
        #     'delayed': res['data']['quote']['delayed'],
        #     'tick_size': res['data']['quote']['tick_size'],
        #     'float_shares': res['data']['quote']['float_shares'],
        #     'limit_down': res['data']['quote']['limit_down'],
        #     'no_profit': res['data']['quote']['no_profit'],
        #     'timestamp_ext': res['data']['quote']['timestamp_ext'],
        #     'lot_size': res['data']['quote']['lot_size'],
        #     'lock_set': res['data']['quote']['lock_set'],
        #     'weighted_voting_rights': res['data']['quote']['weighted_voting_rights'],
        #     'eps': res['data']['quote']['eps'],
        #     'profit_four': res['data']['quote']['profit_four'],
        #     'volume_ratio': res['data']['quote']['volume_ratio'],
        #     'profit_forecast': res['data']['quote']['profit_forecast'],
        #     'low52w': res['data']['quote']['low52w'],
        #     'exchange': res['data']['quote']['exchange'],
        #     'pe_forecast': res['data']['quote']['pe_forecast'],
        #     'total_shares': res['data']['quote']['total_shares'],
        #     'status': res['data']['quote']['status'],
        #     'is_vie_desc': res['data']['quote'].get('is_vie_desc'),
        #     'security_status': res['data']['quote'].get('security_status'),
        #     'goodwill_in_net_assets': res['data']['quote']['goodwill_in_net_assets'],
        #     'weighted_voting_rights_desc': res['data']['quote']['weighted_voting_rights_desc'],
        #     'is_vie': res['data']['quote'].get('is_vie'),
        #     'issue_date': res['data']['quote']['issue_date'],
        #     'sub_type': res['data']['quote']['sub_type'],
        #     'is_registration_desc': res['data']['quote']['is_registration_desc'],
        #     'no_profit_desc': res['data']['quote']['no_profit_desc'],
        #     'dividend': res['data']['quote']['dividend'],
        #     'dividend_yield': res['data']['quote']['dividend_yield'],
        #     'currency': res['data']['quote']['currency'],
        #     'navps': res['data']['quote']['navps'],
        #     'profit': res['data']['quote']['profit'],
        #     'pe_lyr': res['data']['quote']['pe_lyr'],
        #     'pledge_ratio': res['data']['quote']['pledge_ratio'],
        #     'is_registration': res['data']['quote']['is_registration'],
        #     'pb': res['data']['quote']['pb'],
        #     'limit_up': res['data']['quote']['limit_up'],
        #     'pe_ttm': res['data']['quote']['pe_ttm'],
        #     # 其他信息
        #     'pankou_ratio': res['data']['others']['pankou_ratio'],
        #     'cyb_switch': res['data']['others']['cyb_switch']
        # }
        # data = {
        #     'trade_volume': res['data'][0]['trade_volume'],
        #     'side': res['data'][0]['side'],
        #     'is_trade': res['data'][0]['is_trade'],
        #     'level': res['data'][0]['level'],
        #     'trade_session': res['data'][0]['trade_session'],
        #     'trade_type': res['data'][0]['trade_type'],
        #     'trade_unique_id': res['data'][0]['trade_unique_id'],
        #     'bid_appl_seq_num': res['data'][0]['bid_appl_seq_num'],
        #     'offer_appl_seq_num': res['data'][0]['offer_appl_seq_num'],
        #     'trade_type_v2': res['data'][0]['trade_type_v2'],
        #     'yield_to_maturity': res['data'][0]['yield_to_maturity']
        # }
        if sync_es:
            id = '_'.join([str(data[id]) for id in ["mr_code","t"]])
            today = datetime.today().strftime("%Y%m%d")
            self.esLib.client.index(index=f"snowball_ssjy_{today}",document=data,id=id)
        return data

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
    def setup_router(self):
        self.router = APIRouter()
        @self.router.get("/start_ssjy")
        async def start():
            scheduler = BackgroundScheduler()
 
            codes=['全志科技','瑞芯微','欧菲光',"永辉超市","乐鑫科技","联创电子","万达信息","银邦股份","蒙草生态","拉卡拉",
           "新华传媒","宗申动力","隆基绿能","常山北明","旗天科技","国泰君安",
           "国新健康","普洛药业","隆基绿能","中船应急","福日电子","立讯精密","华力创通","中粮资本",
           "东方财富","中国中免","国金证券",
           "中际旭创","小商品城","源杰科技",
           "新元科技","金三江","海达股份","科创新源","华盛锂电","矩阵股份","民德电子","帝尔激光","宇邦新材","乾照光电",
           "保变电气","新诺威","珠江啤酒","国电电力","协创数据","神宇股份","北新建材","未名医药","蜂助手","如通股份",
           "锐捷网络","吉比特","宁德时代","迈为股份","中熔电气","同花顺","光智科技","韦尔股份",
           "上证指数","深证成指","创业板指","中证500"
           ]
            kwargs = {
                "func":self.quotec_detail,
                "codes":codes,
                "sync_es":True
            }
            scheduler.add_job(self.parallel_execute, misfire_grace_time=6,trigger=CronTrigger(hour='9',minute='20-59',second='*/3'),kwargs=kwargs)
            scheduler.add_job(self.parallel_execute, misfire_grace_time=6,trigger=CronTrigger(hour='10',minute='00-59',second='*/3'),kwargs=kwargs)
            scheduler.add_job(self.parallel_execute, misfire_grace_time=6,trigger=CronTrigger(hour='11',minute='00-30',second='*/3'),kwargs=kwargs)
            scheduler.add_job(self.parallel_execute, misfire_grace_time=6,trigger=CronTrigger(hour='13-14',minute='00-59',second='*/3'),kwargs=kwargs)
            scheduler.add_job(self.parallel_execute, misfire_grace_time=6,trigger=CronTrigger(hour='15',minute='00-20',second='*/3'),kwargs=kwargs)
            scheduler.add_job(self.parallel_execute, misfire_grace_time=6,trigger=CronTrigger(hour='9',minute='30-59',second='*/3'),kwargs=kwargs)
            scheduler.start()   
            return {"message":"服务已启动！"}
        @self.router.get("/sql/{sql_str}")
        async def sql(sql_str):
            data = self.esLib.sql(sql_str)
            return {"sql":sql_str,"data":data}
        return self.router