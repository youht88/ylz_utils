import json
import re
from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.data import StringLib,Spinner

def query_neo4j(neo4jLib:Neo4jLib,vars,query,only_data=False):
    spinner = Spinner()
    try:
        query_variables =re.findall(r"\$(.+?)\s",query)
        kwargs = {}
        for key in query_variables:
            if key in vars:
                kwargs[key] = vars[key]
        spinner.start()
        results = neo4jLib.query(query,only_data,**kwargs)
        spinner.end()
        return results
    except Exception as e:
        spinner.end()      
        raise e
    
def neo4j_test(args):
    user = args.user
    password = args.password
    host = args.host
    neo4jLib = Neo4jLib(host,user,password)
    print("*"*50,"let's start","*"*50)
    idx = 0
    vars = {}
    while True:
        idx +=1
        query = input(f'{StringLib.color(f"输入语句{idx}: ")}')
        if query.strip().lower().startswith('/'):
            command =  query.strip().lower().split(" ")[0]
            if command not in ["/q","/set","/get","/list","/clear","/load","/query"]:
                print(
"""
usage:
    /set <key>=<value> 设置变量
    /get <key> 查看key变量
    /list 变量列表
    /clear 清除变量
    /load <file name> as <key> 读取文件内容设置为变量key
    /query <key>=<CQL查询语句> 将查询语句的结果设置为变量key
    /q 退出
"""
                )
                continue
        if query.strip().lower() == '/q':
            print("Good bye!")
            break
        if query.strip().startswith('/set '):
            kv = query.strip().replace("/set","").strip().split('=')
            try:
                key =kv[0].strip()
                value = kv[1].strip()
                vars[key] = json.loads(value)
            except Exception as e:
                print(e)
            continue
        if query.strip().startswith('/get '):
            key = query.strip().replace("/get","").strip()
            if key in vars:
                print(f"{key}={vars[key]}")
            continue
        if query.strip().lower().startswith('/list'):
            print(list(vars.keys()))
            continue
        if query.strip().lower().startswith('/clear'):
            vars = {}
            continue
        if query.strip().startswith('/query '):
            try:
                commands = query.strip().replace('/query ',"").split("=",1)
                key = commands[0].strip()
                query = commands[1].strip()            
                records = query_neo4j(neo4jLib,vars,query,True)
                vars[key] = records
            except Exception as e:
                print(StringLib.lred("ERROR ON QUERY:"),e)
            continue
        if query.strip().startswith('/load'):
            try:
                commands = query.strip().replace('/load',"").split(" as ")
                filename = commands[0].strip()
                key = commands[1].strip()
                with open(filename,"r") as f:
                    text = f.read()
                    vars[key] = json.loads(text)
            except Exception as e:
                print(StringLib.lred("ERROR ON LOAD:"),e)
            continue
        try:
            records,summary,keys = query_neo4j(neo4jLib,vars,query,False)
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
        print("\n")

    neo4jLib.close()