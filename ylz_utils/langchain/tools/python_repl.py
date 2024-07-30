from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

class PythonREPLTool():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def get_tool(self,name=None):
        # You can create the tool to pass to an agent
        python_repl = PythonREPL()
        name = name or "python_repl"
        repl_tool = Tool(
            name=name,
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        )
        return repl_tool