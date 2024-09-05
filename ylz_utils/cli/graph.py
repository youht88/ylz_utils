from ylz_utils.data import StringLib
from ylz_utils.langchain import LangchainLib
from langchain_core.messages import HumanMessage,ToolMessage
from ylz_utils.langchain.graph.db_graph import DbGraph
from ylz_utils.langchain.graph.engineer_graph import EngineerGraph
from ylz_utils.langchain.graph.life_graph import LifeGraph
from ylz_utils.langchain.graph.self_rag_graph import SelfRagGraph
from ylz_utils.langchain.graph.stand_graph import StandGraph

def input_with_readline(prompt):
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print("输入的内容存在编码问题，请确保使用 UTF-8 编码的字符，并不要使用回退键。")
        return input_with_readline(prompt)
    
def start_graph(langchainLib:LangchainLib,args):
    # # 设置标准输入为 UTF-8 编码
    # sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())
    graph_key = args.graph or 'stand'
    llm_key = args.llm_key
    llm_model = args.llm_model
    message = args.message
    chat_dbname = args.chat_dbname
    rag_indexname = args.rag_indexname
    query_dbname = args.query_dbname
    user_id = args.user or 'default'
    conversation_id = args.conversation or 'default'
    thread_id = f"{user_id}-{conversation_id}"
    websearch_key = args.websearch
    graphLib = None
    match graph_key:
        case "stand":
            graphLib = StandGraph(langchainLib)
        case "life":
            graphLib = LifeGraph(langchainLib)
        case "engineer":
            graphLib = EngineerGraph(langchainLib)
        case "db":
            graphLib = DbGraph(langchainLib)
        case "selfrag":
            graphLib = SelfRagGraph(langchainLib)
        case _ :
            return
    if chat_dbname:                                  
        graphLib.set_chat_dbname(chat_dbname)
        print("!!!",f"使用对话数据库{chat_dbname}")
    if rag_indexname:
        retriever = langchainLib.vectorstoreLib.get_store_with_provider_and_indexname(rag_indexname).as_retriever()
        graphLib.set_ragsearch_tool(retriever)
        print("!!!",f"使用知识库{rag_indexname}")
    if query_dbname:
        if query_dbname=="Chinook.db":
            print(StringLib.color(
                "执行wget https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db,确保Chinook.db在当前目录。",
                ["lmagenta","underline"]))
        graphLib.set_query_dbname(query_dbname)
        print("!!!",f"使用查询数据库{query_dbname}")
    if websearch_key:
        graphLib.set_websearch_tool(websearch_key)
        print("!!!",f"使用搜索工具{websearch_key}")
    graphLib.set_nodes_llm_config((llm_key,llm_model))
    graphLib.set_thread(user_id,conversation_id)
    graph = graphLib.get_graph()
#     system_message = \
# """
# 请始终使用中文，并确保中文正确。
# 要求如下：
#  - 一步一步解决问题，过程中进行反思以达到最佳回复
#  - 只有计算和时间问题才使用python_repl工具,使用python_repl工具时要记得用print函数返回结果。陈述事实的时候请不要使用工具
#  - 不要产生幻觉，不知道的问题优先从互联网查询，关于用户自己的问题可以向用户询问。
#  - 当碰到需要需要用户来确认的问题或你需要用户告诉你的问题时，请使用使用human工具向用户询问。注意，询问时请提出询问的具体问题，不要重复我提出的问题
# """
    system_message = None
    while True:
        if not message:
            #message = input("User Input: ")
            message = input_with_readline("User Input: ")
        else:
            print(f"User:{message}")
        if message.lower() in ["/quit", "/exit", "/stop","/q","/bye"]:
            print("Goodbye!")
            break
        #system_message = """请始终用中文回答"""
        if message=="@@NONE@@":
            message = None
        graphLib.graph_stream(graph,message,thread_id = thread_id,system_message=system_message)
        system_message=None
        if graphLib.human_action(graph,thread_id):
            message = "@@NONE@@"
            continue
        message = ""

    current_state = graphLib.graph_get_state(graph,thread_id)
    print("\n本次对话的所有消息:\n",current_state.values["messages"])

    #langchainLib.graphLib.export_graph(graph)
