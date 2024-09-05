from langchain_core.messages import ToolMessage

from .state import State,RequestAssistance
from .node import Node

class HumanNode(Node):
    def __call__(self,state: State):
        score = state["score"]
        score.score=0
        return {"score":score}