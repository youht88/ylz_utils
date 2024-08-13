from typing import TypedDict,Annotated
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list,add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str   
