
import asyncio
import logging
import argparse

from ylz_utils.cli.init import init
from ylz_utils.cli.start import start
from ylz_utils.cli.serve import serve

def run():
    main()

def main():
    parser = argparse.ArgumentParser(description = "测试工具")
    parser.add_argument("--log_level",type=str,default="INFO",choices=["INFO","DEBUG"],help="日志级别,默认:INFO")
    parser.add_argument("--log",type=str,default="task.log",help="日志文件名称")
    parser.add_argument("--env_file",type=str,required=False,help="配置文件名称")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="可以使用的子命令")

    start_parser = subparsers.add_parser("start", help="启动测试")
    start_parser.add_argument("--mode",type=str,
                              required=True,
                              choices=["llm","prompt","loader","runnable","tools","rag","outputParser","graph","other"],
                              help="测试内容")
    start_parser.add_argument("--llm",type=str,help="语言模型标识，例如：LLM.DEEPSEEK")
    # start_parser.add_argument("--crawl",type=int,default=0,choices=[0,1,2,3],help="爬取网页的层级深度,默认:0,表示仅当前网页")
    # start_parser.add_argument("--only_download",type=bool,default=False,help="仅下载网页html,不进行翻译。默认:False (json模式该参数不起作用)")
    # start_parser.add_argument("-s","--size",type=int,default=1500,help="切分文件的字节大小,默认:1500")
    # start_parser.add_argument("-c","--clear_error",action="store_true",help="清除task.json文件中的错误信息,默认:False")

    serve_parser = subparsers.add_parser("serve", help="启动langserve")
    serve_parser.add_argument("--host",type=str,default="0.0.0.0",help="bind host,default:0.0.0.0")
    serve_parser.add_argument("--port",type=int,default=8000,help="listen port,default::8000")
    serve_parser.add_argument("--path",type=str,default="/test",help="path,default:: /test")
    
    args = parser.parse_args()

    init(args)

    if args.command == "start":
        asyncio.run(start(args))
    elif args.command == "serve":
        serve(args)
    else:
        print("未知的命令")
# python3 ../../fix.py --mode json fixDict --dict_hash b614b4 --new_text="一条消息由消息头和消息体组成。以下部分专注于消息体结构。有关消息头结 构，请参阅："
# python3 ../../fix.py --mode json fixDict --old_text "abcde" --new_text="ABCDE"
# python3 ../../fix.py --mode json clearTask --url_id d2a41fe3fc36fe7e998e88623d2889a8 --blocks 1 3 5
# python3 ../../fix.py --mode json fixDict --old_text "<.+>(.*消息由.*?)</.+>" --new_text "报文由报文头和报文正文组成。以下部 分专注于报文正文结构。报文头结构请参阅" --issubtext
# python3 ../../fix.py --mode json fixDict --old_text "<.+>(.*0s.*?)</.+>" --new_text "" --issubtext -l

if __name__ == "__main__":
   main()
