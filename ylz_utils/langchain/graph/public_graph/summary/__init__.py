from typing import Literal
from ylz_utils.langchain.graph import GraphLib
from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage,RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode,tools_condition

from .state import State

class SummaryGraph(GraphLib):
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
    def get_graph(self):
        workflow = StateGraph(State)
        # Define the conversation node and the summarize node
        workflow.add_node("conversation", self.call_model)
        workflow.add_node(self.summarize_conversation)

        # Set the entrypoint as conversation
        workflow.add_edge(START, "conversation")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `conversation`.
            # This means these are the edges taken after the `conversation` node is called.
            "conversation",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
        )

        # We now add a normal edge from `summarize_conversation` to END.
        # This means that after `summarize_conversation` is called, we end.
        workflow.add_edge("summarize_conversation", END)

        # Finally, we compile it!
        graph = workflow.compile(checkpointer=self.memory)
        return graph
    
    def human_action(self, graph, config=None ,thread_id=None):
        return super().human_action(graph, config,thread_id)
    
    
    def call_model(self,state: State,config:RunnableConfig):
        llm_key = config["configurable"].get("llm_key")
        llm_model = config["configurable"].get("llm_model")
        if llm_key or llm_model:
                llm = self.langchainLib.get_llm(llm_key,llm_model)
        else:
                llm = self.get_node_llm()
        # If a summary exists, we add this in as a system message
        summary = state.get("summary", "")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
        response = llm.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    def should_continue(self,state: State) -> Literal["summarize_conversation", "__end__"]:
        """Return the next node to execute."""
        messages = state["messages"]
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
        # Otherwise we can just end
        return END
    def summarize_conversation(self,state: State,config: RunnableConfig):
        llm_key = config["configurable"].get("llm_key")
        llm_model = config["configurable"].get("llm_model")
        if llm_key or llm_model:
            llm = self.langchainLib.get_llm(llm_key,llm_model)
        else:
            llm = self.get_node_llm()

        # First, we summarize the conversation
        print("State keys:",state.keys())
        summary = state.get("summary", "")
        if summary:
            # If a summary already exists, we use a different system prompt
            # to summarize it than if one didn't
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = llm.invoke(messages)
        # We now need to delete messages that we no longer want to show up
        # I will delete all but the last two messages, but you can change this
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
