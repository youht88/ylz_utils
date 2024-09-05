from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.life_graph import LifeGraph

import jionlp
import time
import datetime

class Node():
    def __init__(self,lifeGraph:LifeGraph,msg=None):
        self.graphLib = lifeGraph
        if msg:
            print(msg)
        
    def parse_time(self,sdt,edt,duration):
        parse_sdt=None
        parse_edt=None
        try:
            parse_sdt = jionlp.parse_time(sdt,time_base=time.time())
            print("sdt:",parse_sdt)
        except Exception as e:
            pass
        try:
            parse_edt = jionlp.parse_time(edt,tiem_base=time.time())    
            print("edt:",parse_edt)
        except Exception as e:
            pass
        
        if not parse_sdt and not parse_edt:
            sdt = datetime.datetime.now()
            edt = datetime.datetime.now()
        elif parse_sdt and not parse_edt:
            sdt = datetime.datetime.fromisoformat(parse_sdt['time'][0])
            edt = datetime.datetime.fromisoformat(parse_sdt['time'][1])
        elif parse_edt and not parse_sdt:
            sdt = datetime.datetime.fromisoformat(parse_edt['time'][0])
            edt = datetime.datetime.fromisoformat(parse_edt['time'][1])
        elif parse_edt and parse_sdt:
            sdt = datetime.datetime.fromisoformat(parse_sdt['time'][0])
            edt = datetime.datetime.fromisoformat(parse_edt['time'][1])
        return sdt,edt,duration