from datetime import datetime, timedelta
import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from .mairui_base import MairuiBase

class HIMK(MairuiBase):
    def __init__(self):
        super().__init__()
        self.register_router()

    def get_himk_ltszph(self,sync_es:bool=False):   
        """获取沪深A股流通市值排名"""
        #数据更新：每天20:00（约10分钟完成更新）
        #请求频率：1分钟20次 
        today = datetime.today()
        yestoday = datetime.today() - timedelta(days=1)
        if today.hour>21:
            ud = today.strftime("%Y-%m-%d")
        else:
            ud = yestoday.strftime("%Y-%m-%d")
        add_fields = {"ud":ud}
        date_fields = ['t','ud']
        skip_condition = f"(ud.dt.strftime('%Y-%m-%d')>='{ud}')"
        keys=['dm','ud']
        name = f"himk_ltszph"
        
        df = self._load_data(name,f"himk/ltszph",
                            add_fields=add_fields,
                            skip_condition=skip_condition,
                            keys=keys,date_fields=date_fields)
        if sync_es:
            es_result = self.esLib.save(name,df,ids=['dm','ud'])
            print(f"errors:{es_result["errors"]}")
        return df
    def register_router(self):
        @self.router.get("/himk/ltszph",response_class=HTMLResponse)
        async def get_himk_ltszph(req:Request):
            """获取沪深A股流通市值排名"""
            try:
                df = self.get_himk_ltszph()
                magic = req.query_params.get('magic')
                if magic!=None:
                    df = df[df['c'].apply(self.is_magic)] 
                    df = df.reset_index(drop=True)   
                df = self._prepare_df(df,req)
                content = df.to_html()
                return HTMLResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"{e}")