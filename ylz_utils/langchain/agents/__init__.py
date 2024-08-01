from langgraph.prebuilt import chat_agent_executor

class AgentLib():
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
    def get_agent(self,llm,tools):
        return chat_agent_executor.create_function_calling_executor(llm,tools)
