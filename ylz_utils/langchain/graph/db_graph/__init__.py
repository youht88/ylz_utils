from ylz_utils.langchain.graph import GraphLib

from langgraph.graph import StateGraph, START,END, MessagesState
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from langchain_community.utilities import SQLDatabase
from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode

from langchain_core.tools import tool

from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.prompts import ChatPromptTemplate


from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

class DbGraph(GraphLib):
    db = None
    def __init__(self,langchainLib):
        super().__init__(langchainLib)
    def set_db(self,uri:str):
        self.db = SQLDatabase.from_uri(uri)
    def _check(self):
        if not self.db:
            raise Exception("先调用set_db(uri)设置db")
        if not self.llm:
            raise Exception("先调用set_llm(key,model)设置llm")
    def set_toolkit(self):
        self._check()
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        self.list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        self.get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    @tool
    def db_query_tool(self,query: str) -> str:
        """
        Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """
        self._check()
        result = self.db.run_no_throw(query)
        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        return result
   
    def create_tool_node_with_fallback(self,tools: list) -> RunnableWithFallbacks[Any, dict]:
        """
        Create a ToolNode with a fallback to handle errors and surface them to the agent.
        """
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )
    def handle_tool_error(self,state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }
    def first_tool_call(self,state: State) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_list_tables",
                            "args": {},
                            "id": "tool_abcd123",
                        }
                    ],
                )
            ]
        }
    def model_check_query(self,state: State) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        self._check()
        query_check_system = """You are a SQL expert with a strong attention to detail.
        Double check the SQLite query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

        You will call the appropriate tool to execute the query after running this check."""

        query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", query_check_system), ("placeholder", "{messages}")]
        )
        query_check = query_check_prompt | self.llm.bind_tools(
            [self.db_query_tool], tool_choice="required"
        )
        return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

    def query_gen_node(self,state: State):
        self._check()
        # Add a node for a model to generate a query based on the question and schema
        query_gen_system = """You are a SQL expert with a strong attention to detail.

        Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

        DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

        When generating the query:

        Output the SQL query that answers the input question without a tool call.

        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.

        If you get an error while executing a query, rewrite the query and try again.

        If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
        NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

        If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

        query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", query_gen_system), ("placeholder", "{messages}")]
        )
        query_gen = query_gen_prompt | self.llm.bind_tools(
            [SubmitFinalAnswer]
        )
        message = query_gen.invoke(state)

        # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            tool_messages = []
        return {"messages": [message] + tool_messages}
    def should_continue(self,state: State) -> Literal[END, "correct_query", "query_gen"]:
        messages = state["messages"]
        last_message = messages[-1]
        # If there is a tool call, then we finish
        if getattr(last_message, "tool_calls", None):
            return END
        if last_message.content.startswith("Error:"):
            return "query_gen"
        else:
            return "correct_query"
        
    def get_graph(self) -> CompiledStateGraph:
        # Define a new graph
        self._check()
        self.set_toolkit()
        
        workflow = StateGraph(State)
        workflow.add_node("first_tool_call", self.first_tool_call)
        workflow.add_node(
            "list_tables_tool", self.create_tool_node_with_fallback([self.list_tables_tool])
        )
        workflow.add_node("get_schema_tool", self.create_tool_node_with_fallback([self.get_schema_tool]))
        model_get_schema = self.llm.bind_tools(
            [self.get_schema_tool]
        )
        workflow.add_node(
            "model_get_schema",
            lambda state: {
                "messages": [model_get_schema.invoke(state["messages"])],
            },
        )
        workflow.add_node("query_gen", self.query_gen_node)
        workflow.add_node("correct_query", self.model_check_query)
        workflow.add_node("execute_query", self.create_tool_node_with_fallback([self.db_query_tool]))

        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_conditional_edges(
            "query_gen",
            self.should_continue,
        )
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")

        graph = workflow.compile(checkpointer=self.memory)
        return graph 
    