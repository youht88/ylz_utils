from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph.stand_graph import StandGraph

class Node():
    def __init__(self,standGraph:StandGraph,msg=None):
        self.graphLib = standGraph
        if msg:
            print(msg)
