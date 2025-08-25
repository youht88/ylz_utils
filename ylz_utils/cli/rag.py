from langchain_community.document_loaders import UnstructuredFileLoader
from tqdm import tqdm
import time
from ylz_utils import LangchainLib
from langchain_community.vectorstores import FAISS

from langgraph.prebuilt import create_react_agent

from rich import print
from rich.markdown import Markdown
from rich.console import Console

from langchain_core.messages import AIMessage,HumanMessage,ToolMessage

async def start_rag(langchainLib:LangchainLib,args):
    embedding_key = args["embedding_key"]
    embedding_model = args["embedding_model"]
    rag_key = args["rag_key"] or "chroma"
    rag_indexname = args["rag_indexname"] or "index"
    rag_metadata = args["rag_metadata"] or {}
    rag_filter = args["rag_filter"] or {}
    url = args["url"]
    pptx = args["pptx"]
    docx = args["docx"]
    pdf = args["pdf"]
    txt = args["txt"]
    glob = args["glob"]
    message = args["message"]
    fake_size = args["fake_size"]
    batch = args["batch"]
    rag_method = args["rag_method"] or "skip"
    
    llm_key = args["llm_key"]
    llm_model = args["llm_model"]

    if (not url and not pptx and not docx and not pdf and not txt and not glob) and  (not message):
        print(f"1、指定url/pptx/docx/pdf/txt/glob:系统将文档下载切片后向量化到{rag_indexname}数据表\n2、指定message:系统将从{rag_indexname}数据表中搜索相关记录。\n您需要至少指定url/pptx/docx/pdf/txt/glob和message中的一个参数.")
        return

    if embedding_key or embedding_model or fake_size:
        embedding = langchainLib.get_embedding(embedding_key,embedding_model,fake_size=fake_size)
    else:
        embedding = None

    
    vectorstoreLib = langchainLib.get_vectorstoreLib(rag_key)
    vectorstore = vectorstoreLib.get_store(rag_indexname,embedding=embedding)

    if (url or pptx or docx or pdf or txt or glob):
        docs = start_loader(langchainLib,args)
        print("#"*60)
        if not docs:
            print("没有找到文档")
            return

        # 存储到向量数据库         
        ids = vectorstoreLib.add_docs(vectorstore=vectorstore,docs=docs,batch=batch,metadata_filter=rag_filter,method=rag_method)
        print(f"add docs use `{rag_method}` method and total ids:",len(ids))   
    
        if isinstance(vectorstore,FAISS):
            vectorstoreLib.save(vectorstore)
    
    ### 执行查找任务
    if message and rag_indexname:           
        # print("检索记录如下--->",langchainLib.vectorstoreLib.search_with_score(message,vectorstore,metadata_filter=rag_filter,k=4))
        # print("all done!!")
        llm = langchainLib.get_llm(llm_key,llm_model)
        def rag_docs(query):
            '''根据要查询的query获得rag文档'''
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4,"filter":rag_filter})
            return retriever.invoke(query)
        agent = create_react_agent(llm,[rag_docs])
        console = Console()
        # for chunk in agent.stream({"messages":[("user",message)]},stream_mode="values"):
        #     console.print(chunk,style="bold red")
        async for event in agent.astream_events({"messages":[("user",message)]},version="v2"):
            kind = event.get("event")
            #console.print(f"{kind},{event.keys()}",style="bold red")
            if kind == 'on_chat_model_stream':
                print(event['data']['chunk'].content,end="",flush=True)
                #console.print("assistant:",Markdown(event['data']['chunk'].content),style="bold green")
            elif kind == 'on_chat_model_start':
                print("assistant:",end="",flush=True)
            elif kind == 'on_chat_model_end':
                print()
            if kind  == 'on_retriever_start':
                console.print(f"获取参考数据,tags:{event['tags']}",style="bold blue")
            elif kind  == 'on_retriever_end':
                console.print(f"获取{len(event['data']['output'])}份相关文档",style="blue")
            elif kind  == 'on_tool_start':
                console.print(f"调用函数:{event['name']}({event['data']['input']})",style="bold yellow")             
            elif kind  == 'on_tool_end':
                pass

            # if message.get("agent"):
            #     console.print("assistant:",Markdown(message["agent"]["messages"][-1].content),style="bold green")
            # elif isinstance(message,HumanMessage):
            #     console.print("user:",Markdown(message.content),style="bold lightblue")
            # elif message.get("tools"):
            #     console.print("tool:",Markdown(message["tools"]["messages"][-1].content),style="bold yellow")
            # else:
            #     print(message)
        print("all done!!")
    ###### have bug when poetry add sentence_transformers   
    #v1 = langchainLib.get_huggingface_embedding()
    #print("huggingface BGE:",v1)

    ###### tet google embdeeding
    # embed = langchainLib.get_embedding("EMBEDDING.GEMINI")
    # docs = [Document("I am a student"),Document("who to go to china"),Document("this is a table")]
    # vectorestore = langchainLib.vectorstoreLib.faissLib.create_from_docs(docs,embedding=embed)
    # langchainLib.vectorstoreLib.faissLib.save("test.faiss",vectorestore,index_name="gemini")

def start_loader(langchainLib:LangchainLib,args):
    url = args["url"]
    depth = args["depth"]
    docx_file = args["docx"]
    pptx_file = args["pptx"]
    pdf_file = args["pdf"]
    txt_file = args["txt"]
    glob:str = args["glob"]
    rag_metadata = args["rag_metadata"] or {}
    chunk_size  = args["rag_size"] or 512
    only_one = list(filter(lambda x: x,[url,docx_file,pptx_file,pdf_file,txt_file,glob]))
    result = []
    if len(only_one) != 1:
        print("请指定url,docx,pptx,pdf,txt,glob其中的一个")
        return 
    if url:
        documentLib = langchainLib.get_documentLib("url")
        result = documentLib.load_and_split(url = url,metadata=rag_metadata,max_depth = depth, chunk_size=chunk_size)
    elif docx_file:
        documentLib = langchainLib.get_documentLib("docx")
        result = documentLib.load_and_split(docx_file,metadata=rag_metadata,chunk_size=chunk_size)
    elif pptx_file:
        documentLib = langchainLib.get_documentLib("pptx")
        result = documentLib.load_and_split(pptx_file,metadata=rag_metadata,chunk_size=chunk_size)
    elif pdf_file:
        documentLib = langchainLib.get_documentLib("pdf")
        result = documentLib.load_and_split(pdf_file,metadata=rag_metadata,chunk_size=chunk_size)
    elif txt_file:
        documentLib = langchainLib.get_documentLib("txt")
        result = documentLib.load_and_split(txt_file,metadata=rag_metadata,chunk_size=chunk_size)
    elif glob:
        if glob.find(".docx")>=0:
            documentLib = langchainLib.get_documentLib("docx")
        elif glob.find(".pptx")>=0:
            documentLib = langchainLib.get_documentLib("pptx")
        elif glob.find(".pdf")>=0:
            documentLib = langchainLib.get_documentLib("pdf")
        elif glob.find(".txt")>=0:
            documentLib = langchainLib.get_documentLib("txt")
        else:
            documentLib = langchainLib.get_documentLib("txt")
        
        result  = documentLib.dirload_and_split(".",glob=glob,metadata=rag_metadata,chunk_size=chunk_size)
    
    if result:  
        print("total doc chunk:",len(result))
        print("first doc chunk:",result[0])
    return result
