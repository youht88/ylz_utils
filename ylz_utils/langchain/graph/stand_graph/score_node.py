from .state import State,RequestAssistance
from .node import Node
from langchain_core.runnables import RunnablePassthrough

class ScoreNode(Node):
    def __init__(self,standGraph,msg=None):
        super().__init__(standGraph,msg)
        self.llm = self.graphLib.get_node_llm()

    def __call__(self,state:State):
        messages = state["messages"]
        prompt = self.graphLib.langchainLib.get_prompt(
"""
1、对以下问题评估是否需要询问用户，如果需要给出要询问的内容，如果可以直接回答或需要查询互联网或使用工具则score=1
2、使用python_repl工具时，所执行command脚本必须有return返回值,例如`'command': 'return 3 + 2'`
""",human_keys={},human_prompt="",
        use_chat=True)
        #llm = self.langchainLib.get_llm(model = "llama3-groq-70b-8192-tool-use-preview")
        #llm = self.langchainLib.get_llm(model = "llama3-70b-8192")
        chain = {"history":RunnablePassthrough()} | prompt | self.llm.with_structured_output(RequestAssistance)
        response:RequestAssistance = chain.invoke(messages)

        print(response)
        return {"score":response}
        