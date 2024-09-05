import json
import re
from ylz_utils.file import IOLib
from typing import Literal
from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.data import StringLib,Spinner
from ylz_utils.langchain import LangchainLib
from langchain_core.messages import AIMessage,HumanMessage
from ylz_utils.langchain.graph.life_graph import LifeGraph
from ylz_utils.langchain.graph.stand_graph import StandGraph

def query_neo4j(neo4jLib:Neo4jLib,vars,query):
    spinner = Spinner()
    try:
        query_variables =re.findall(r"\$(.+?)\s",query)
        kwargs = {}
        for key in query_variables:
            if key in vars:
                kwargs[key] = vars[key]
        spinner.start()
        result = neo4jLib.query(query,**kwargs)
        spinner.end()
        return result
    except Exception as e:
        spinner.end()      
        raise e
    
def neo4j_test(args):
    user = args.user
    password = args.password
    host = args.host
    user_id = args.user_id or 'default'
    conversation_id = args.conversation_id or 'default'
    thread_id = f"{user_id}-{conversation_id}"
    dbname = args.chat_dbname or 'chat.sqlite'
    llm_key = args.llm_key
    llm_model =args.llm_model
    embedding_key = args.embedding_key
    langchainLib = LangchainLib()
    llm = langchainLib.get_llm(llm_key)
    embedding = langchainLib.get_embedding(embedding_key)
    
    prompt = langchainLib.get_prompt(use_chat=True)
    langchainLib.llmLib.set_dbname(dbname)
    chat = langchainLib.get_chat(llm,prompt)
    
    neo4jLib = Neo4jLib(host,user,password)
    langchainLib.init_neo4j(neo4jLib)

    lifeGraph = None
    life_graph = None
    standGraph = None
    stand_graph = None

    print("*"*50,"let's start","*"*50)
    idx = 0
    vars = {}
    history = []
    mode:Literal['neo4j','llm','life','stand','test'] = 'neo4j'
    query=""

    while True:
        idx +=1
        if not query:
            query = IOLib.input_with_history(f'{StringLib.color(f"输入语句{idx}:")}',history)
        #history.append(query)
        if query.strip().lower().startswith('/'):
            command =  query.strip().lower().split(" ")[0]
            if command not in ["/q","/set","/get","/list","/clear","/load","/query","/import","/history","/mode"]:
                print(
"""
usage:
    <CQL> 执行CQL语句
    /set <key>=<value> 设置变量
    /get <key> 查看key变量
    /list 变量列表
    /clear 清除变量
    /load <file name> as <key> 读取文件内容设置为变量key
    /query <key>=<CQL查询语句> 将查询语句的结果设置为变量key
    /import <import.json> 批量创建object/subject节点，以及他们直接的relationship
        创建节点根据nodes以<node_label>_key为唯一标识。
            - 以merge方式创建nodes
            - nodes的第一行属性作为节点属性,每一行都应保持同样的结构
            - <node_label>_key必须指定
        创建关系根据relations,为(from:from_label{...})-[type]->(to:to_label{...})关系表
            - 如果relations存在则merge方式创建relation
            - 每一行结构必须包含from,from_label字段和可选的from_key
            - 第一行属性(除from,to,from_label,to_label,from_key,to_key,type外)作为关系属性
    /mode <neo4j|llm|life|stand|test >  
        - neo4j Neo4j REPL
        - llm <message> 不带graph的对话
        - life 带life_graph的对话 
        - stand 带stand_graph的对话
        - test 测试graph对话
    /q 退出
"""
                )
            elif query.strip().lower() == '/q':
                print("Good bye!")
                break
            elif query.strip().startswith('/set '):
                kv = query.strip().replace("/set","").strip().split('=')
                try:
                    key =kv[0].strip()
                    value = kv[1].strip()
                    vars[key] = json.loads(value)
                except Exception as e:
                    print(e)

            elif query.strip().startswith('/get '):
                key = query.strip().replace("/get","").strip()
                if key in vars:
                    print(f"{key}={vars[key]}")

            elif query.strip().lower().startswith('/list'):
                print(list(vars.keys()))

            elif query.strip().lower().startswith('/clear'):
                vars = {}

            elif query.strip().startswith('/query '):
                try:
                    commands = query.strip().replace('/query ',"").split("=",1)
                    key = commands[0].strip()
                    query = commands[1].strip()            
                    result = query_neo4j(neo4jLib,vars,query)
                    vars[key] = neo4jLib.get_data(result.records)
                except Exception as e:
                    print(StringLib.lred("ERROR ON QUERY:"),e)

            elif query.strip().startswith('/load'):
                try:
                    commands = query.strip().replace('/load',"").split(" as ")
                    filename = commands[0].strip()
                    key = commands[1].strip()
                    with open(filename,"r") as f:
                        text = f.read()
                        vars[key] = json.loads(text)
                except Exception as e:
                    print(StringLib.lred("ERROR ON LOAD:"),e)

            elif query.strip().startswith('/import'):
                try:
                    filename = query.strip().replace('/import',"").strip()
                    with open(filename,"r") as f:
                        importer = json.load(f)
                        neo4jLib.create_node_and_relation(importer)
                except Exception as e:
                    print(StringLib.lred("ERROR ON IMPORT:"),e)

            elif query.strip().startswith('/history'):
                try:
                    command = query.strip().replace('/history',"").strip()
                    match command:
                        case "clear":
                            history = []
                            idx=0
                        case "":
                            for i,item in enumerate(history):
                                print(f"{i+1} - {item}")
                        case _:
                            if  command.isdigit():
                                goto_idx = int(command) 
                            if goto_idx > 0 and goto_idx <= len(history):
                                query = history[goto_idx - 1]
                                continue
                            else: 
                                all = [(i+1,item) for (i,item) in enumerate(history) if re.findall(command,item)]
                                for i,item in all:
                                    print(f"{i} - {item}")
                except Exception as e:
                    print(StringLib.lred("ERROR ON HISTORY:"),e)

            elif query.strip().startswith('/mode'):
                command = query.strip().replace('/mode',"").strip().lower()
                if command not in ['llm','life','test','neo4j']:
                    print("必须指定llm,life,test,neo4j其中的一种!")                
                else:
                    mode = command
                print('当前mode:',mode)
        elif mode == 'llm':
            spinner = Spinner()
            try:
                command = query
                spinner.start()
                res = chat.stream({"input":command}, {'configurable': {'user_id': user_id,'conversation_id': conversation_id}} )
                spinner.end()
                print("AI: ",end="")
                for chunk in res:
                    print(chunk.content,end='')
                print("")
            except Exception as e:
                spinner.end()
                print(StringLib.lred("ERROR ON LLM:"),e)

        elif mode == 'life':
            if not life_graph:
                lifeGraph = LifeGraph(langchainLib)
                lifeGraph.set_nodes_llm_config((llm_key,None))
                lifeGraph.set_thread(user_id,conversation_id)
                life_graph = lifeGraph.get_graph()

            spinner = Spinner()
            try:
                command = query
                spinner.start()
                res = lifeGraph.graph_stream(life_graph,command,thread_id = thread_id)
                spinner.end()
            except Exception as e:
                spinner.end()
                print(StringLib.lred("ERROR ON LLM:"),e)
        elif mode == 'stand' :
            if not stand_graph:
                standGraph = StandGraph(langchainLib)
                standGraph.set_nodes_llm_config((llm_key,None))
                standGraph.set_thread(user_id,conversation_id)
                stand_graph = standGraph.get_graph()

            spinner = Spinner()
            try:
                if query=="@@NONE@@":
                    command = None
                else:
                    command = query
                spinner.start()
                res = standGraph.graph_stream(stand_graph,command,thread_id = thread_id)
                spinner.end()
                if standGraph.human_action(stand_graph,thread_id):
                    query = "@@NONE@@"
                    continue
            except Exception as e:
                spinner.end()
                print(StringLib.lred("ERROR ON Test:"),e)
        else:
            try:
                records,summary,keys = query_neo4j(neo4jLib,vars,query)
                # Records
                for record in records[:5]:
                    print(record.data())
                if records[5:]:
                    n = len(records) - 5
                    if n > 0:
                        print(".........")
                        for record in records[-min(n,5):]:
                            print(record.data())
                # Summary information
                if summary:
                    print("The query `{query}` returned {records_count} records in {time} ms.".format(
                        query=StringLib.color(summary.query), records_count=StringLib.color(len(records)),
                        time=StringLib.color(summary.result_available_after)
                    ))
            except Exception as e:
                print(StringLib.lred("ERROR ON QUERY:"),e)
        
        query=""

    neo4jLib.close()