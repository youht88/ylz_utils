from pydantic import BaseModel,Field
from typing import TypedDict,Annotated

from langgraph.graph import add_messages

class RequestAssistance(BaseModel):
    """
    Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str =Field(description="当涉及个人状况的时候，使用礼貌的用语向用户提出明确的要询问的问题")
    score:int = Field(description="对询问该问题的必要性给出0-5的评分",min=0,max=5)  

class State(TypedDict):
    messages: Annotated[list,add_messages]
    score: RequestAssistance

