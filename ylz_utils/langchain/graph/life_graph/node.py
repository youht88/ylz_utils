from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.life_graph import LifeGraph

import jionlp
import time
import datetime

class Node():
    neo4jLib = None
    def __init__(self,lifeGraph:LifeGraph):
        self.lifeGraph = lifeGraph
        self.jionlp = jionlp
        self.get_neo4jLib()
        
    def get_neo4jLib(self):
        neo4jLib =  self.lifeGraph.langchainLib.neo4jLib
        if not neo4jLib:
            raise Exception("请先调用langchainLib.init_neo4j(neo4j)")
        self.neo4jLib = neo4jLib
        return neo4jLib
    
    def get_llm(self,llm_key=None,llm_model=None):
        llm_key = llm_key or self.lifeGraph.llm_key
        llm_model = llm_model or self.lifeGraph.llm_model
        print("LLM to Used:",llm_key,llm_model)
        return self.lifeGraph.langchainLib.get_llm(llm_key,llm_model)
    
    def parse_time(self,sdt,edt,duration):
        parse_sdt=None
        parse_edt=None
        try:
            parse_sdt = self.jionlp.parse_time(sdt,time_base=time.time())
            print("sdt:",parse_sdt)
        except Exception as e:
            pass
        try:
            parse_edt = self.jionlp.parse_time(edt,tiem_base=time.time())    
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