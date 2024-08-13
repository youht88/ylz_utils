from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

class WolframAlphaTool():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def get_tool(self):
        WOLFRAM_ALPHA_APPID = "U83YPW-GR2VEAQJV5"
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_APPID)
        return wolfram