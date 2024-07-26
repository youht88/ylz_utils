
import asyncio
import logging
import argparse

from ylz_utils.cli.init import init

from ylz_utils.cli.reset import reset
from ylz_utils.cli.start import start
from ylz_utils.cli.serve import serve

def run():
    main()

def main():
    parser = argparse.ArgumentParser(description = "测试工具")
    parser.add_argument("--project_name",type=str,default="ylz_utils",help="project名称")
    parser.add_argument("--config_name",type=str,default="config.yaml",help="config名称")
    parser.add_argument("--log_level",type=str,default="INFO",choices=["INFO","DEBUG"],help="日志级别,默认:INFO")
    parser.add_argument("--log_name",type=str,default="ylz_utils.log",help="日志文件名称")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="可以使用的子命令")
    
    reset_parser = subparsers.add_parser("reset", help="执行初始化")

    start_parser = subparsers.add_parser("start", help="启动测试")
    start_parser.add_argument("--mode",type=str,
                              required=True,
                              choices=["llm","chat","prompt","loader","runnable","tools","rag","outputParser","graph","other"],
                              help="测试内容")
    start_parser.add_argument("--llm",type=str,help="语言模型标识，例如：LLM.DEEPSEEK")
    start_parser.add_argument("--model",type=str,help="model")
    start_parser.add_argument("--message",type=str,default="hello",help="input message")
    start_parser.add_argument("--user",type=str,help="user_id,example: alice")    
    start_parser.add_argument("--conversation",type=str,help="conversation_id,example: 123") 

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
           
    args = parser.parse_args()

    #print("args====>",args)

    init(args)
   
    if args.command == "reset":
        reset(args)
    elif args.command == "start":
        asyncio.run(start(args))
    elif args.command == "serve":
        serve(args)

if __name__ == "__main__":
   main()
