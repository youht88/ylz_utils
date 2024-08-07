from ylz_utils import Color, StringLib
from ylz_utils import FileLib
from ylz_utils import LangchainLib

from pydantic import BaseModel,Field
from typing import Literal,List
from pptx.util import Inches,Cm,Pt
from pptx.enum.shapes import MSO_SHAPE

from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import UnstructuredFileLoader

def __agent(langchainLib:LangchainLib,args):
    llm_key = args.llm_key
    llm_model = args.llm_model
    message = args.message if args.message else "厦门今天天气怎么样?"
    llm = langchainLib.get_llm(llm_key,llm_model)
    prompt = langchainLib.get_prompt()
    tools = []
    tavily_tool = langchainLib.toolLib.web_search.get_tool()
    tools.append(tavily_tool)
    # langchain_docs_vectorestore = langchainLib.vectorstoreLib.faiss.load("langchain_docs.db")
    # langchain_docs_retriever = langchain_docs_vectorestore.as_retriever()
    # rag_tool = langchainLib.toolLib.rag_search.get_tool(langchain_docs_retriever,"langchain docs","langchain的文档")
    # tools.append(rag_tool)
    
    agent =  langchainLib.get_agent(llm,tools)
    res = agent.stream({"messages":[("user",message)]})
    for chunk in res:
        print(chunk , end="")
    print("\n\n",langchainLib.get_llm(full=True),"\n")

def __chat(langchainLib:LangchainLib,args):
    llm_key = args.llm_key
    message = args.message
    model = args.llm_model
    user_id = args.user if args.user else 'default'
    conversation_id = args.conversation if args.conversation else 'default'
    llm = langchainLib.get_llm(key=llm_key,model=model)
    dbname = args.chat_dbname or 'chat.sqlite'
    #### chat 模式
    prompt = langchainLib.get_prompt(use_chat=True)
    if dbname:
        langchainLib.llmLib.set_dbname(dbname)
    chat = langchainLib.get_chat(llm,prompt)
    chain = chat | langchainLib.get_outputParser()
    while True:
        if not message:
            message=input("USER:")
        else:
            print(f"USER:{message}")
        if message.lower() in ['/quit','/exit','/stop','/bye','/q']:
            print("Good bye!!")
            break
        res = chain.stream({"input":message}, {'configurable': {'user_id': user_id,'conversation_id': conversation_id}} )
        message = ""
        print("AI:",end="")
        for chunk in res:
            print(chunk,end="")
        print("\n")
    print("\n",langchainLib.get_llm(full=True))

def __llm_test(langchainLib:LangchainLib,args):
    llm_key = args.llm_key
    for _ in range(3):
        llm = langchainLib.get_llm(llm_key)
        res = llm.invoke("hello")
        print("res:",res)
    print("llms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.get_llm(full=True)])

def __outputParser_test(langchainLib:LangchainLib,args):
    message = args.message
    llm_key = args.llm_key
    class Food(BaseModel):
        name: str = Field(description="name of the food")
        place: str = Field(defualt="未指定",description="where to eat food?",examples=["家","公司","体育馆"])
        calories: float|None = Field(title="卡路里(千卡)",description="食物热量")
        amount: tuple[float,str] = Field(description="tuple of the value and unit of the food amount")
        time: str = Field(default = "未指定",description="the time of eating food,确保每个食物都有对应的时间，如果没有指定则为空字符串")
        health: Literal["健康","不健康","未知"]|None = Field(description="根据常识判断食物是否为健康食品",examples=["健康","不健康","未知"])
    class Foods(BaseModel):
        foods: List[Food]

    # Set up a parser + inject instructions into the prompt template.
    parser = langchainLib.get_outputParser(Foods)

    prompt1 = PromptTemplate(
        template="用中文分析句子的内容。如果没有指定食物热量则根据食物的名称和数量进行估计。判断食物是否为健康\n{format_instructions}\n{command}\n",
        input_variables=["command"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    # And a query intended to prompt a language model to populate the data structure.
    fixed_parser = OutputFixingParser.from_llm(parser=parser, llm=langchainLib.get_llm(llm_key))
    # retry_parser = RetryOutputParser.from_llm(parser=parser, llm=langchainLib.get_llm())

    prompt2 = langchainLib.get_prompt(
            system_prompt = "用中文分析句子的内容。如果没有指定食物热量则根据食物的名称和数量进行估计。同时判断食物是否为健康",
            human_keys={"command":"输入的语句"},
            outputParser = parser,
            use_chat = False
        )

    #chain = prompt2 | langchainLib.get_llm() | fixed_parser
    
    chain = prompt2 | langchainLib.get_llm(llm_key) | parser
    
    # retry_chain = RunnableParallel(
    #     completion=chain, prompt_value=prompt1,
    #     ) | RunnableLambda(lambda x:retry_parser.parse_with_prompt(**x))

    #output = chain.invoke({"command": "我今天在家吃了3个麦当劳炸鸡和一个焦糖布丁，昨天8点在电影院吃了一盘20千卡的西红柿炒蛋"})
    output = chain.invoke({"command": message})
    #retry_parser.parse_with_prompt(output['completion'], output['prompt_value'])
    #retry_parser.parse_with_prompt(output['completion'],output['prompt_value'])
    print("\noutput parser:",output)

    print("llms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.get_llm(full=True)])


def __runnalble_test(langchainLib:LangchainLib):
    runnable1 = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x["num"] + 1,
    other = RunnableParallel(
        passed=RunnablePassthrough(),
        modified=lambda x: x["num"] * 2,
        )
    )
    #runnable1.invoke({"num":1})
    def f(x):
        return x['modified']*100

    runnable2 = runnable1 | RunnableParallel(
        {"origin": RunnablePassthrough(),
        "new1": lambda x: x["other"]["passed"],
        "new2": RunnableLambda(f)}
        )

    #for trunck in runnable2.stream({"num": 1}):
    #  print(trunck)
    res = runnable2.invoke({"num": 1})
    print("\nrunnalbe:",res)

def __prompt_test(langchainLib:LangchainLib,args):
    llm_key = args.llm_key
    llm_model = args.llm_model
    prompt = langchainLib.get_prompt(f"你对日历时间非常精通",human_keys={"context":"关联的上下文","question":"问题是"},use_chat = False)
    prompt.partial(context="我的名字叫小美")   # 这个为什么不起作用?
    print("!!!!",langchainLib.llmLib.default_llm_key,llm_key)
    llm = langchainLib.get_llm(llm_key,llm_model)
    print(llm)
    outputParser = langchainLib.get_outputParser()
    chain = prompt | llm | outputParser

    res = chain.invoke({"question":"我叫什么名字，今天礼拜几","context":"我是海涛"})
    print(res)
    res = chain.stream({"question":"用我的名字写一首诗歌","context":"我的名字叫小美"})
    for chunk in res:
        print(chunk,end="")
    
    prompt = PromptTemplate.from_template("用中文回答：{topic} 的{what}是多少")
    llm = langchainLib.get_llm("LLM.GROQ")
    print(llm)
    chain = prompt | llm | outputParser
    promise = chain.batch([{"topic":"中国","what":"人口"},{"topic":"美国","what":"国土面积"},{"topic":"新加坡","what":"大学"}])
    print(promise)
    print("llms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.get_llm(full=True)])

def __rag_test(langchainLib:LangchainLib,args):
    embedding_key = args.embedding_key
    embedding_model = args.embedding_model
    faiss_dbname = args.embedding_dbname or "embedding.faiss"
    url = args.url
    pptx = args.pptx
    docx = args.docx
    pdf = args.pdf
    glob = args.glob
    message = args.message
    
    
    if (not url and not pptx and not docx and not pdf and not glob) and  (not message):
        print(f"1、指定url/pptx/docx:系统将文档下载切片后向量化到{faiss_dbname}数据库\n2、指定message:系统将从{faiss_dbname}数据库中搜索相关的两条记录。\n您需要至少指定url和message中的一个参数.")
        return

    if embedding_key or embedding_model:
        embedding = langchainLib.get_embedding(embedding_key,embedding_model)
    else:
        embedding = None

    if (url or pptx or docx or pdf or glob):
        docs = __loader_test(langchainLib,args)
        print("#"*60)
        if not docs:
            return
        if url and faiss_dbname:
            ##### create vectorestore
            # url = "https://python.langchain.com/v0.2/docs/concepts/#tools"
            # faiss_dbname = "langchain_docs.faiss"
            print("result:",[{"doc_len":len(doc['doc'].page_content),"doc_blocks":len(doc['blocks'])} for doc in docs])
            for doc in docs:
                blocks = doc['blocks']
                vectorestore,ids = langchainLib.vectorstoreLib.faiss.create_from_docs(blocks,embedding)
                langchainLib.vectorstoreLib.faiss.save(faiss_dbname,vectorestore)
                print("ids:",ids)
        else:
                vectorestore,ids = langchainLib.vectorstoreLib.faiss.create_from_docs(docs,embedding)
                langchainLib.vectorstoreLib.faiss.save(faiss_dbname,vectorestore)
                print("ids:",ids)
    if message and faiss_dbname:   
        # docs = [Document("I am a student"),Document("who to go to china"),Document("this is a table")]
        # vectorestore = langchainLib.vectorstoreLib.faiss.create_from_docs(docs)
        # langchainLib.vectorstoreLib.faiss.save("test.faiss",vectorestore)
        
        vectorestore = langchainLib.vectorstoreLib.faiss.load(faiss_dbname,embedding)
        print("v--->",langchainLib.vectorstoreLib.faiss.search(message,vectorestore,k=2))
    
    ###### have bug when poetry add sentence_transformers   
    #v1 = langchainLib.get_huggingface_embedding()
    #print("huggingface BGE:",v1)

    ###### tet google embdeeding
    # embed = langchainLib.get_embedding("EMBEDDING.GEMINI")
    # docs = [Document("I am a student"),Document("who to go to china"),Document("this is a table")]
    # vectorestore = langchainLib.vectorstoreLib.faiss.create_from_docs(docs,embedding=embed)
    # langchainLib.vectorstoreLib.faiss.save("test.faiss",vectorestore,index_name="gemini")

def __loader_test(langchainLib:LangchainLib,args):
    url = args.url
    depth = args.depth
    docx_file = args.docx
    pptx_file = args.pptx
    pdf_file = args.pdf
    glob = args.glob
    chunk_size = args.size or 512
    only_one = list(filter(lambda x: x,[url,docx_file,pptx_file,pdf_file,glob]))
    if len(only_one) != 1:
        print(f"请指定url,docx,pptx,pdf,glob其中的一个")
        return 
    if url:
        result = langchainLib.load_url_and_split_markdown(url = url,max_depth = depth, chunk_size=chunk_size)
        print("result:",[{"doc_len":len(doc['doc'].page_content),"doc_blocks":len(doc['blocks']),"metadata":doc['metadata']} for doc in result])
    elif docx_file:
        result = langchainLib.documentLib.docx.load_and_split(docx_file,chunk_size=chunk_size)
        print(result)
    elif pptx_file:
        result = langchainLib.documentLib.pptx.load_and_split(pptx_file,chunk_size=chunk_size)
        print(result)
    elif pdf_file:
        result = langchainLib.documentLib.pdf.load_and_split(pdf_file,chunk_size=chunk_size)
        print(result)
    elif glob:
        loader_cls = UnstructuredFileLoader
        if glob.find(".docx")>=0:
            loader_cls = langchainLib.documentLib.docx.loader
        elif glob.find(".pptx")>=0:
            loader_cls = langchainLib.documentLib.pptx.loader
        elif glob.find(".pdf")>=0:
            loader_cls = langchainLib.documentLib.pdf.loader
        loader  = langchainLib.documentLib.dir.loader(".",glob=glob,show_progress=True,loader_cls=loader_cls)
        result  = loader.load_and_split(langchainLib.splitterLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=0))
        print(result)
    return result

def __tools_test(langchainLib:LangchainLib,args):
    tool = langchainLib.toolLib.python_repl.get_tool()
    res = tool.invoke(
"""
print(3+2)
""")
    print("python repl=",res)
    tool = langchainLib.toolLib.wolfram_alpha.get_tool()
    res = tool.run("what is the square root of 25?")
    print(res)
    return 
    # tool: TavilySearchResults = langchainLib.get_search_tool("TAVILY")
    # prompt = langchainLib.get_prompt(is_chat=False,human_keys={"context":"关联的上下文是:","question":"问题是:"})
    # res = tool.invoke("易联众现在股价是多少？")
    # print(res)
    # llm = langchainLib.get_llm("LLM.DEEPSEEK")
    # chain = RunnableParallel({
    #     "question": RunnablePassthrough(),
    #     "context": tool
    # }) | prompt | llm 
    # res = chain.invoke("易联众现在股价是多少？")
    # print(res)

    # print("#"*50)
    class Output(BaseModel):
        name:str = Field(description= "姓名")
        age:int = Field(description= "年龄")

    langchainLib.add_plugins()
    llm_key = args.llm_key
    llm = langchainLib.get_llm(llm_key)
    print("llms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.get_llm(full=True)])

    outputParser = langchainLib.get_outputParser(Output,fix=True,llm = llm,retry=3)
    prompt = langchainLib.get_prompt(human_keys = {"ask":"问题"},use_chat=False)
    search = langchainLib.get_search_tool("TAVILY")
    tools = [search]
    #chain  = prompt | (lambda x: x.messages[-1].content) | search | (lambda x:'```json\n{"name":中国,"age":"20"}') | outputParser
    #chain  = prompt | llm | outputParser
    chain = prompt | RunnableLambda(lambda x:x.text,name="抽取text") | search 
    #print("chain to_json = ",chain.to_json())
    #print("chain name=",chain.get_name())
    print("\nchain graph=",chain.get_graph(),"\n")
    res = chain.invoke({"ask": "北京人口多少"})
    print(type(res),res)
    #FileLib.writeFile("graph1.png",chain.get_graph().draw_mermaid_png(),mode="wb")

    # llm = langchainLib.get_llm("LLM.DEEPSEEK")
    # print("llm",llm.to_json())
    llm.bind_tools(tools)
    res = llm.invoke({"ask":"北京2024年人口多少"})
    print(res)
def __graph_test(langchainLib:LangchainLib,args):
    llm_key = args.llm_key
    llm_model = args.llm_model
    message = args.message
    dbname = args.chat_dbname
    faiss_dbname = args.embedding_dbname
    user = args.user or 'default'
    conversation = args.conversation or 'default'
    thread_id = f"{user}-{conversation}"
    websearch_key = args.websearch
    if dbname:                                  
        langchainLib.graphLib.set_dbname(dbname)
    if faiss_dbname:
        retriever = langchainLib.vectorstoreLib.faiss.load("embedding.faiss").as_retriever()
        langchainLib.graphLib.set_ragsearch_tool(retriever)
        print("!!!",f"使用知识库{faiss_dbname}")
    if websearch_key:
        langchainLib.graphLib.set_websearch_tool(websearch_key)
        print("!!!",f"使用搜索工具{websearch_key}")
        
    graph = langchainLib.get_graph(llm_key=llm_key,llm_model=llm_model)
    if not message:
        message = input("User: ")
    langchainLib.graphLib.graph_stream(graph,message,thread_id = thread_id,
                                    system_message="所有问题务必请用中文回答,如果没有能力回答请升级为专家模式。只有计算和时间问题才使用python_repl工具") 

    
    # while True:
    #     if not message:
    #         message = input("User: ")
    #     else:
    #         print(f"User:{message}")
    #     if message.lower() in ["/quit", "/exit", "/stop","/q","/bye"]:
    #         print("Goodbye!")
    #         break
    #     langchainLib.graphLib.graph_stream(graph,message,thread_id = thread_id)
    #     message=""
    
    # while True:
    #     current_state = langchainLib.graphLib.graph_get_state(graph,thread_id)
        
    #     if current_state.values["messages"] and current_state.values["messages"][-1].content=="":        
    #         langchainLib.graphLib.graph_stream(graph,None,thread_id = thread_id) 
    #     elif current_state.values["messages"]:
    #         if not message:
    #             message = input("User: ")
    #         if not message or message == '/q':
    #             break
    #     else:
    #         if not message:
    #             message = input("User: ")
    #         if not message or message == '/q':
    #             break
    #         langchainLib.graphLib.graph_stream(graph,message,thread_id = thread_id,
    #                                         system_message="所有问题务必请用中文回答,如果没有能力回答请升级为专家模式。只有计算和时间问题才使用python_repl工具") 
    #     message = ""
        # print("*"*50)  
        # langchainLib.graphLib.graph_get_state_history(graph,thread_id=thread_id)
        # print("*"*50)
        # langchainLib.graphLib.graph_get_state(graph,thread_id=thread_id)
    current_state = langchainLib.graphLib.graph_get_state(graph,thread_id)
    print("\n\n",current_state.values["messages"], "Next: ", current_state.next)

    langchainLib.graphLib.export_graph(graph)

def start(args):
    langchainLib = LangchainLib()
    #langchainLib.add_plugins()
    if args.mode == "llm":
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试llm {Color.RESET}")
        __llm_test(langchainLib,args)
    elif args.mode == "prompt":
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试prompt {Color.RESET}")
        __prompt_test(langchainLib,args)    
    elif args.mode == 'loader':    
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试loader {Color.RESET}")
        __loader_test(langchainLib,args)    
    elif args.mode == 'runnable':
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试runnable {Color.RESET}")
        __runnalble_test(langchainLib)
    elif args.mode == 'outputParser':    
        StringLib.logging_in_box(f"{Color.YELLOW} 测试outputParser {Color.RESET}")
        __outputParser_test(langchainLib,args)
    elif args.mode == 'rag':    
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试rag {Color.RESET}")
        __rag_test(langchainLib,args)
    elif args.mode == 'tools':
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试tools {Color.RESET}")
        __tools_test(langchainLib,args)
    elif args.mode == 'graph':
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试graph {Color.RESET}")
        __graph_test(langchainLib,args)
    elif args.mode == 'chat':
        __chat(langchainLib,args)
    elif args.mode == 'agent':
        __agent(langchainLib,args)
    else:
        print("args=",args)
        print("llms--->:",[(item["type"],item["api_key"],item["model"],item["used"]) for item in langchainLib.get_llm(full=True)])
        print("embeddings---->:",[(item["type"],item["api_key"],item["model"],item["used"]) for item in langchainLib.get_embedding(full=True)])
        #loader = langchainLib.documentLib.pptx.loader("30335320.pptx")
        #docs = loader.load()
        #print(docs)
        return
        loader = langchainLib.documentLib.pptx.newer("test.pptx")
        loader.add_slide(0).set_title("Hello World","gogogo").add_text("youht",10,10,60,40)
        tab = [{"name":"youht","age":20},{"name":"jinli","age":10}] 
        tx = loader.add_slide(1).set_title("你好","step1").add_text("Step1",10,10,100,40)
        loader.add_text_paragraph(tx,"如何学习python",size=30,level=1)
        loader.add_text_paragraph(tx,"numpy",bold=True,size=20,level=2)
        loader.add_slide(2).set_title("你好","step1").add_table(tab,100,100,600,400,with_header = True)
        shape = loader.add_slide(8).set_title("Shape").add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,50,50,50,50,
                                                        line_brightness=0.5,
                                                        text="hello")
        print(loader.get_shape_sizes(shape))
        loader.add_slide().set_title("chart1").add_chart("pie",
                                                        [{"s1":{"a":1,"b":2,"c":3}},
                                                         {"s2":{"a":3,"b":2,"c":1}}],10,10,600,300)
        loader.add_slide().set_title("chart2").add_chart("bubble",
                                                        [{"s1":[(1,2,10),(4,6,3),(2,3,1)]},
                                                         {"s2":[(2,1,2),(8,2,3),(4,1,5)]}],10,10,600,300)

        loader.save()