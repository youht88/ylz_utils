from ylz_utils.langchain import LangchainLib
from langchain_core.messages import HumanMessage,ToolMessage

def start_graph(langchainLib:LangchainLib,args):
    llm_key = args.llm_key
    llm_model = args.llm_model
    message = args.message
    chat_dbname = args.chat_dbname
    rag_dbname = args.rag_dbname
    user = args.user or 'default'
    conversation = args.conversation or 'default'
    thread_id = f"{user}-{conversation}"
    websearch_key = args.websearch
    if chat_dbname:                                  
        langchainLib.graphLib.set_dbname(chat_dbname)
        print("!!!",f"使用对话数据库{chat_dbname}")
    if rag_dbname:
        retriever = langchainLib.vectorstoreLib.faiss.load(rag_dbname).as_retriever()
        langchainLib.graphLib.set_ragsearch_tool(retriever)
        print("!!!",f"使用知识库{rag_dbname}")
    if websearch_key:
        langchainLib.graphLib.set_websearch_tool(websearch_key)
        print("!!!",f"使用搜索工具{websearch_key}")
        
    graph = langchainLib.get_graph(llm_key=llm_key,llm_model=llm_model)
    system_message = \
"""
请确保中文正确。只有计算和时间问题才使用python_repl工具,使用python_repl工具时要记得用print函数返回结果
不要产生幻觉，不知道的问题优先从互联网查询，关于用户自己的问题可以向用户询问。
当碰到需要需要用户来确认的问题或你需要用户告诉你的问题时，请使用使用human工具向用户询问。注意，询问时请提出询问的具体问题，不要重复我提出的问题
"""
    while True:
        if not message:
            message = input("User: ")
        else:
            print(f"User:{message}")
        if message.lower() in ["/quit", "/exit", "/stop","/q","/bye"]:
            print("Goodbye!")
            break
        #system_message = """请始终用中文回答"""
        langchainLib.graphLib.graph_stream(graph,message,thread_id = thread_id,system_message=system_message)
        snapshot = langchainLib.graphLib.graph_get_state(graph,thread_id)
        if snapshot.next: 
            question = snapshot.values["messages"][-1].tool_calls[0]["args"]["request"]            
            tool_call_id = snapshot.values["messages"][-1].tool_calls[0]["id"]
            message = input(f"{question}: ")
            if message.strip().lower() != '':
                tool_message = ToolMessage(tool_call_id=tool_call_id, content=message)
                langchainLib.graphLib.graph_update_state(graph,thread_id=thread_id, values = {"messages":[tool_message]})
                langchainLib.graphLib.graph_stream(graph,None,thread_id = thread_id)

        message = ""

    current_state = langchainLib.graphLib.graph_get_state(graph,thread_id)
    print("\n本次对话的所有消息:\n",current_state.values["messages"])

    langchainLib.graphLib.export_graph(graph)
