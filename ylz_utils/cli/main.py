import click
import json
from typing import Optional
from datetime import datetime

from ylz_utils.cli.set_logger import set_logger

from ylz_utils.cli.neo4j import neo4j_test as click_neo4j_test
from ylz_utils.cli.init import init as click_init
from ylz_utils.cli.start import start as click_start
from ylz_utils.cli.serve import serve as click_serve

def validate_json(ctx, param, value):
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise click.BadParameter(f"必须是有效的JSON格式")
    return None

def run():
    main()

@click.group()
@click.option("--project_name",type=str,default="ylz_utils",help="project名称")
@click.option("--config_name",type=str,default="config.yaml",help="config名称")
@click.option("--log_level",default="INFO",type=click.Choice(["INFO","DEBUG"]),help="日志级别,默认:INFO")
@click.option("--log_name",type=str,default="ylz_utils.log",help="日志文件名称")
@click.pass_context
def main(ctx,**kwargs):
    """ylz_utils命令行工具"""
    ctx.obj = kwargs
    set_logger(ctx.obj)

@main.command()
@click.pass_context
def init(ctx):
    """初始化配置信息"""
    click_init(ctx.obj)

@main.command()
@click.option("--mode",
                required=True,
                type=click.Choice(["llm","chat","prompt","loader","runnable","tools","rag","outputParser","graph","agent","info"]),
                help="测试内容")
@click.option("--llm_key",type=str,help="语言模型标识，例如：LLM.DEEPSEEK")
@click.option("--embedding_key",type=str,help="嵌入模型标识，例如：EMBEDDING.TOGETHER")
@click.option("--llm_model",type=str,help="语言模型model")
@click.option("--embedding_model",type=str,help="嵌入模型model")
@click.option("--message",type=str,help="input message")
@click.option("--user_id",type=str,help="user_id,example: alice")    
@click.option("--conversation_id",type=str,help="conversation_id,example: 123") 
@click.option("--url",type=str,help="仅rag,loader使用,下载的URL地址")    
@click.option("--depth",type=int,default=1,help="仅rag使用,下载的深度，默认为1")
@click.option("--rag_key",type=click.Choice(["es","faiss","chroma","pg"]),help="向量数据库") 
@click.option("--rag_indexname",type=str,help="保存的向量索引表,格式为indexname") 
@click.option("--rag_metadata",type=str,callback=validate_json,help="保存rag文档的metadata,格式为json") 
@click.option("--rag_method",type=click.Choice(["replace","skip","add"]),default="skip",help="添加相同的rag文档时的处理方式,默认skip") 
@click.option("--rag_filter",type=str,callback=validate_json,help="查询rag文档的filter,格式为json") 
@click.option("--rag_size",type=int,default=512,help="文档分隔的size,默认512字节") 
@click.option("--chat_dbname",type=str,help="保存的对话数据库") 
@click.option("--query_dbname",default='Chinook.db',type=str,help="测试查询的数据库，默认Chinook.db") 
@click.option("--docx",type=str,help="docx文档文件名") 
@click.option("--pptx",type=str,help="pptx文档文件名") 
@click.option("--pdf",type=str,help="pdf文档文件名") 
@click.option("--txt",type=str,help="txt文档文件名") 
@click.option("--glob",type=str,help="当前目录下的glob匹配的文件")
@click.option("--websearch",type=click.Choice(["tavily","duckduckgo","serpapi"]),help="websearch的工具")   
@click.option("--graph",type=click.Choice(["stand","life","engineer","db","selfrag","test","stock"]),help="内置graph的类型") 
@click.option("--fake_size",type=int,help="使用fake embeding的size，当fake_size>0是使用fake embeding，并且维度为fake_size") 
@click.option("--batch",type=int,default=10,help="使用生成embeding时的以batch为度量显示进度，默认分隔为10批") 

@click.pass_context
def start(ctx, **kwargs):
    click_start(kwargs)
             
@main.command()
@click.option("--host",type=str,default="0.0.0.0",help="bind host,default:0.0.0.0")
@click.option("--port",type=int,default=8000,help="listen port,default::8000")
@click.option("--path",type=str,default="/test",help="path,default:: /test")
@click.option("--llm_key",type=str,help="llm provider,example: LLM.DEEPSEEK")    
@click.option("--llm_model",type=str,help="llm model,example: deepseek-chat") 
@click.option("--embedding_key",type=str,help="embedding provider,example: LLM.DASHSCOPE")    
@click.option("--embedding_model",type=str,help="embedding model,example: deepseek-chat") 
@click.option("--user_id",type=str,help="user_id,example: alice")    
@click.option("--conversation_id",type=str,help="conversation_id,example: 123") 
@click.option("--rag_indexname",type=str,help="保存的向量索引表,格式为<es|faiss|chroma>:<indexname>") 
@click.pass_context
def serve(ctx,**kwargs):
    click_serve(kwargs)

if __name__ == '__main__':
    main()
