from ylz_utils.database.neo4j import Neo4jLib
from ylz_utils.data import StringLib,Spinner
def neo4j_test(args):
    user = args.user
    password = args.password
    host = args.host
    neo4jLib = Neo4jLib(host,user,password)
    print("*"*50,"连接成功!","*"*50)
    idx = 0
    while True:
        spinner = Spinner()
        idx +=1
        query = input(f'输入语句{idx}: ')
        if query.strip().lower() == '/q':
            print("Good bye!")
            break
        try:
            spinner.start()
            records,summary,keys = neo4jLib.query(query)
            spinner.end()
            # Records
            for record in records:
                print(record)
            
            # Summary information
            print("The query `{query}` returned {records_count} records in {time} ms.".format(
                query=StringLib.color(summary.query), records_count=StringLib.color(len(records)),
                time=StringLib.color(summary.result_available_after)
            ))
        except Exception as e:
            print(e)
            spinner.end()
        print("\n")
