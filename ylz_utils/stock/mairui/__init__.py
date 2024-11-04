from fastapi import FastAPI

from ylz_utils.stock.mairui.mairui_base import MairuiBase
from ylz_utils.stock.mairui.mairui_hibk import HIBK
from ylz_utils.stock.mairui.mairui_higg import HIGG
from ylz_utils.stock.mairui.mairui_himk import HIMK
from ylz_utils.stock.mairui.mairui_hilh import HILH
from ylz_utils.stock.mairui.mairui_hitc import HITC
from ylz_utils.stock.mairui.mairui_hizj import HIZJ
from ylz_utils.stock.mairui.mairui_hscp import HSCP
from ylz_utils.stock.mairui.mairui_hslt import HSLT
from ylz_utils.stock.mairui.mairui_hsmy import HSMY
from ylz_utils.stock.mairui.mairui_hsrl import HSRL
from ylz_utils.stock.mairui.mairui_hszb import HSZB
from ylz_utils.stock.mairui.mairui_hszg import HSZG
from ylz_utils.stock.mairui.mairui_zs import ZS

class MairuiLib():    
    def __init__(self):
        self.hibk = HIBK()
        self.higg = HIGG()
        self.himk = HIMK()
        self.hitc = HITC()
        self.hilh = HILH()
        self.hizj = HIZJ()
        self.hscp = HSCP()
        self.hslt = HSLT()
        self.hsmy = HSMY()
        self.hsrl = HSRL()
        self.hszb = HSZB()
        self.hszg = HSZG()
        self.zs = ZS()     

    def register_app(self,app:FastAPI): 
        base = MairuiBase()
        base.register_router()
        app.include_router(base.router,prefix="/mairui")
        app.include_router(self.higg.router,prefix="/mairui")
        app.include_router(self.himk.router,prefix="/mairui")
        app.include_router(self.hsrl.router,prefix="/mairui")
        app.include_router(self.hilh.router,prefix="/mairui")
        app.include_router(self.hizj.router,prefix="/mairui")
        app.include_router(self.hszg.router,prefix="/mairui")

      

        
