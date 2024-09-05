
import asyncio
import logging
import argparse
from argparse import Namespace

from ylz_utils.cli.init import init

from ylz_utils.cli.neo4j import neo4j_test
from ylz_utils.cli.reset import reset
from ylz_utils.cli.start import start
from ylz_utils.cli.serve import serve

def run():
    main()

def main():
    usage= \
"""
    examples:
        # 初始化配置信息 
        ylz_utils reset 

        # 启动大语言模型对话
        ylz_utils start --mode chat
        
        # 测试neo4j
        ylz_utils neo4j 
"""
    parser = argparse.ArgumentParser(description = "测试工具",usage=usage)
    parser.add_argument("--project_name",type=str,default="ylz_utils",help="project名称")
    parser.add_argument("--config_name",type=str,default="config.yaml",help="config名称")
    parser.add_argument("--log_level",type=str,default="INFO",choices=["INFO","DEBUG"],help="日志级别,默认:INFO")
    parser.add_argument("--log_name",type=str,default="ylz_utils.log",help="日志文件名称")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="可以使用的子命令")
    
    reset_parser = subparsers.add_parser("reset", help="执行初始化")
    custom_service_parser = subparsers.add_parser("custom_service", help="客服example")
    
    neo4j_parser = subparsers.add_parser("neo4j", help="测试neo4j")
    neo4j_parser.add_argument("--user",type=str,help="user")
    neo4j_parser.add_argument("--password",type=str,help="password")
    neo4j_parser.add_argument("--host",type=str,help="host")
    neo4j_parser.add_argument("--llm_key",type=str,help="llm_key")
    neo4j_parser.add_argument("--llm_model",type=str,help="llm_model")
    neo4j_parser.add_argument("--embedding_key",type=str,help="embedding_key")
    neo4j_parser.add_argument("--user_id",type=str,help="user_id,example: alice")
    neo4j_parser.add_argument("--conversation_id",type=str,help="conversation_id,example: 123") 
    neo4j_parser.add_argument("--chat_dbname",type=str,help="保存的对话数据库") 

    start_parser = subparsers.add_parser("start", help="启动测试")
    start_parser.add_argument("--mode",type=str,
                              required=True,
                              choices=["llm","chat","prompt","loader","runnable","tools","rag","outputParser","graph","agent","info"],
                              help="测试内容")
    start_parser.add_argument("--llm_key",type=str,help="语言模型标识，例如：LLM.DEEPSEEK")
    start_parser.add_argument("--embedding_key",type=str,help="嵌入模型标识，例如：EMBEDDING.TOGETHER")
    start_parser.add_argument("--llm_model",type=str,help="语言模型model")
    start_parser.add_argument("--embedding_model",type=str,help="嵌入模型model")
    start_parser.add_argument("--message",type=str,help="input message")
    start_parser.add_argument("--user",type=str,help="user_id,example: alice")    
    start_parser.add_argument("--conversation",type=str,help="conversation_id,example: 123") 
    start_parser.add_argument("--url",type=str,help="仅rag,loader使用,下载的URL地址")    
    start_parser.add_argument("--depth",type=int,default=1,help="仅rag使用,下载的深度，默认为1")
    start_parser.add_argument("--rag_indexname",type=str,help="保存的向量索引表,格式为<es|faiss|chroma>:<indexname>") 
    start_parser.add_argument("--chat_dbname",type=str,help="保存的对话数据库") 
    start_parser.add_argument("--query_dbname",default='Chinook.db',type=str,help="测试查询的数据库，默认Chinook.db") 
    start_parser.add_argument("--docx",type=str,help="docx文档文件名") 
    start_parser.add_argument("--pptx",type=str,help="pptx文档文件名") 
    start_parser.add_argument("--pdf",type=str,help="pdf文档文件名") 
    start_parser.add_argument("--glob",type=str,help="当前目录下的glob匹配的文件")
    start_parser.add_argument("--websearch",type=str,choices=["tavily","duckduckgo"],help="websearch的工具")
     
    
    start_parser.add_argument("--size",type=int,help="文档分隔的size") 
    start_parser.add_argument("--graph",type=str,choices=["stand","life","engineer","db","selfrag"],help="内置graph的类型") 
    start_parser.add_argument("--fake_size",type=int,help="使用fake embeding的size，当fake_size>0是使用fake embeding，并且维度为fake_size") 
    start_parser.add_argument("--batch",type=int,default=10,help="使用生成embeding时的以batch为度量显示进度，默认分隔为10批") 
     

    # start_parser.add_argument("--only_download",type=bool,default=False,help="仅下载网页html,不进行翻译。默认:False (json模式该参数不起作用)")
    # start_parser.add_argument("-s","--size",type=int,default=1500,help="切分文件的字节大小,默认:1500")
    # start_parser.add_argument("-c","--clear_error",action="store_true",help="清除task.json文件中的错误信息,默认:False")

    serve_parser = subparsers.add_parser("serve", help="启动langserve")
    serve_parser.add_argument("--host",type=str,default="0.0.0.0",help="bind host,default:0.0.0.0")
    serve_parser.add_argument("--port",type=int,default=8000,help="listen port,default::8000")
    serve_parser.add_argument("--path",type=str,default="/test",help="path,default:: /test")
    serve_parser.add_argument("--llm",type=str,help="llm,example: LLM.DEEPSEEK")    
    serve_parser.add_argument("--model",type=str,help="model,example: deepseek-chat") 
    serve_parser.add_argument("--user",type=str,help="user_id,example: alice")    
    serve_parser.add_argument("--conversation",type=str,help="conversation_id,example: 123") 
           
    args:Namespace = parser.parse_args()

    #print("args====>",args)

    if args.command =="reset":
        reset(args)
        return
    
    init(args)
   
    if args.command == "start":
        start(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "neo4j":
        neo4j_test(args)
    elif args.command == "custom_service":
        import ylz_utils.cli.custom_service

if __name__ == "__main__":
   main()
