from .state import State,RequestAssistance
from .node import Node

class Score(Node):
    def scoreNode(self,state:State):
        messages = state["messages"]
        prompt = self.standGraph.graphLib.langchainLib.get_prompt(
"""
对以下问题评估是否需要询问用户，如果需要给出要询问的内容，如果不需要询问用户则score=1        
""",
        use_chat=False)
        #llm = self.langchainLib.get_llm(model = "llama3-groq-70b-8192-tool-use-preview")
        #llm = self.langchainLib.get_llm(model = "llama3-70b-8192")
        llm = self.standGraph.llm
        chain = prompt | llm.with_structured_output(RequestAssistance)
        response:RequestAssistance = chain.invoke({"input":messages[-1].content})

        print(response)
        return {"score":response}
        
    def scoreEdge(self,state:State):
        score = state["score"]
        print("scoreEdge",score)
        if score.score <= 1:
            return "chatbotNode"
        else:
            return "humanNode"