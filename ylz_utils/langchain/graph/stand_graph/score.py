from .state import State,RequestAssistance
from .node import Node
from langchain_core.runnables import RunnablePassthrough

class Score(Node):
    def scoreNode(self,state:State):
        self.standGraph.langchainLib.add_plugins(True)
        messages = state["messages"]
        prompt = self.standGraph.langchainLib.get_prompt(
"""
对以下问题评估是否需要询问用户，如果需要给出要询问的内容，如果不需要询问用户则score=1        
""",human_keys={},
        use_chat=True)
        #llm = self.langchainLib.get_llm(model = "llama3-groq-70b-8192-tool-use-preview")
        #llm = self.langchainLib.get_llm(model = "llama3-70b-8192")
        llm = self.standGraph.llm
        chain = {"history":RunnablePassthrough()} | prompt | llm.with_structured_output(RequestAssistance)
        response:RequestAssistance = chain.invoke(messages)

        print(response)
        return {"score":response}
        
    def scoreEdge(self,state:State):
        score = state["score"]
        print("scoreEdge",score)
        if score.score <= 1:
            return "chatbotNode"
        else:
            return "humanNode"