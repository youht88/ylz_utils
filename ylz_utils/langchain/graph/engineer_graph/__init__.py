from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ylz_utils.langchain.graph import GraphLib

class EngineerGraph():
    def __init__(self,graphLib:GraphLib):
        self.graphLib = graphLib