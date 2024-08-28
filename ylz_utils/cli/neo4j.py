import json
import re
from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.data import StringLib,Spinner
def neo4j_test(args):
    user = args.user
    password = args.password
    host = args.host
    neo4jLib = Neo4jLib(host,user,password)
    print("*"*50,"let's start","*"*50)
    idx = 0
    vars = {}
    while True:
        spinner = Spinner()
        idx +=1
        query = input(f'{StringLib.color(f"输入语句{idx}: ")}')
        if query.strip().lower().startswith('/'):
            command =  query.strip().lower().split(" ")[0]
            if command not in ["/q","/set","/get","/list","/clear"]:
                print("usage: /q 退出\n/set <key>=<value> 设置变量\n/get <key> 查看key变量\n/list 变量列表\n/clear 清除变量")
                continue
        if query.strip().lower() == '/q':
            print("Good bye!")
            break
        if query.strip().lower().startswith('/set '):
            kv = query.strip().lower().replace("/set","").strip().split('=')
            try:
                key =kv[0].strip()
                value = kv[1].strip()
                vars[key] = json.loads(value)
            except Exception as e:
                print(e)
            continue
        if query.strip().lower().startswith('/get '):
            key = query.strip().lower().replace("/get","").strip()
            if key in vars:
                print(f"{key}={vars[key]}")
            continue
        if query.strip().lower().startswith('/list'):
            print(list(vars.keys()))
            continue
        if query.strip().lower().startswith('/clear'):
            vars = {}
            continue
        try:
            query_variables =re.findall(r"\$(.+?)\s",query)
            kwargs = {}
            for key in query_variables:
                if key in vars:
                    kwargs[key] = vars[key]
            spinner.start()
            if ";" in query:
                print(query)
                result = neo4jLib.run(query)
                records = result.data
                summary = result._summary
            else:
                records,summary,keys = neo4jLib.query(query,**kwargs)
            spinner.end()
            # Records
            for record in records:
                print(record)
            
            # Summary information
            if summary:
                print("The query `{query}` returned {records_count} records in {time} ms.".format(
                    query=StringLib.color(summary.query), records_count=StringLib.color(len(records)),
                    time=StringLib.color(summary.result_available_after)
                ))
        except Exception as e:
            print(e)
            spinner.end()
        print("\n")

    neo4jLib.close()