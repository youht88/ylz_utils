from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from typing import Type
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel,Field
class PythonREPLArgSchema(BaseModel):
    ''' input str for python repl tool'''
    command: str = Field(description="the command to execute use python repl with `print(...)` at last.")

class PythonREPLTool():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def get_tool(self,name=None):
        # You can create the tool to pass to an agent
        python_repl = PythonREPL()
        name = name or "python_repl"
        #So You are not skilled at dealing with computation issues,All question about calculate shoud use this tool when you get all argument value.
        repl_tool = Tool(
            name=name,
            description=(
              "The tool is a Python shell with print(...) as last. "
              "Use this to execute python commands and get result. Input should be a valid python command. "
              "if you are using finance module,you must know that the argument `Period` of `stock.history` function must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']"
            ),
            args_schema = PythonREPLArgSchema ,
            func=python_repl.run
        )
        return repl_tool

