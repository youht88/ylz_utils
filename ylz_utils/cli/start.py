from ylz_utils.data_utils import Color, StringLib
from ylz_utils.file_utils import FileLib
from ylz_utils.langchain_utils import LangchainLib

from pydantic import BaseModel,Field
from typing import Literal,List

from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser

from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langgraph.graph import START,END,MessageGraph
from langgraph.prebuilt import ToolNode

def __other(langchainLib:LangchainLib,args):
    llm_key = args.llm
    print(langchainLib.get_llm(llm_key))

def __llm_test(langchainLib:LangchainLib,args):
    llm_key = args.llm
    for _ in range(3):
        llm = langchainLib.get_llm(llm_key)
        res = llm.invoke("hello")
        print("res:",res)
    print("llms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.llms])

def __outputParser_test(langchainLib:LangchainLib):
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
    fixed_parser = OutputFixingParser.from_llm(parser=parser, llm=langchainLib.get_llm('LLM.GROQ'))
    # retry_parser = RetryOutputParser.from_llm(parser=parser, llm=langchainLib.get_llm())

    prompt2 = langchainLib.get_prompt(
            system_prompt = "用中文分析句子的内容。如果没有指定食物热量则根据食物的名称和数量进行估计。同时判断食物是否为健康",
            human_keys={"command":"输入的语句"},
            outputParser = parser,
            is_chat = False
        )

    #chain = prompt2 | langchainLib.get_llm() | fixed_parser
    
    chain = prompt2 | langchainLib.get_llm('LLM.GROQ') | parser
    
    # retry_chain = RunnableParallel(
    #     completion=chain, prompt_value=prompt1,
    #     ) | RunnableLambda(lambda x:retry_parser.parse_with_prompt(**x))

    #output = chain.invoke({"command": "我今天在家吃了3个麦当劳炸鸡和一个焦糖布丁，昨天8点在电影院吃了一盘20千卡的西红柿炒蛋"})
    output = chain.invoke({"command": "两个小时前吃了一根雪糕，还喝了一杯咖啡"})
    #retry_parser.parse_with_prompt(output['completion'], output['prompt_value'])
    #retry_parser.parse_with_prompt(output['completion'],output['prompt_value'])
    print("\noutput parser:",output)

    print("\nllms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.llms])


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

async def __prompt_test(langchainLib:LangchainLib):

    prompt = langchainLib.get_prompt(f"你对日历时间非常精通",human_keys={"context":"关联的上下文","question":"问题是"},is_chat = True)
    prompt.partial(context="我的名字叫小美")   # 这个为什么不起作用?
    llm = langchainLib.get_llm("LLM.TOGETHER")
    outputParser = langchainLib.get_outputParser()
    chain = prompt | llm | outputParser

    res = await chain.ainvoke({"question":"我叫什么名字，今天礼拜几","context":"我是海涛"})
    print(res)
    res = chain.stream({"question":"用我的名字写一周诗歌","context":"我的名字叫小美"})
    for chunk in res:
        print(chunk,end="")
    
    prompt = PromptTemplate.from_template("用中文回答：{topic} 的{what}是多少")
    chain = prompt | langchainLib.get_llm("LLM.GROQ") | outputParser
    promise = chain.batch([{"topic":"中国","what":"人口"},{"topic":"美国","what":"国土面积"},{"topic":"新加坡","what":"大学"}])
    print(promise)
    print("\n\nllms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.llms])

def __rag_test(langchainLib:LangchainLib):
    ###### create vectorestore
    # docs = [Document("I am a student"),Document("who to go to china"),Document("this is a table")]
    # vectorestore = langchainLib.create_faiss_from_docs(docs)
    # langchainLib.save_faiss("faiss.db",vectorestore)
    
    vectorestore = langchainLib.load_faiss("faiss.db")
    print("v--->",langchainLib.search_faiss("I want to buy a table?",vectorestore,k=1))
    
    ###### have bug when poetry add sentence_transformers   
    #v1 = langchainLib.get_huggingface_embedding()
    #print("huggingface BGE:",v1)

    ###### tet google embdeeding
    # embed = langchainLib.get_embedding("EMBEDDING.GEMINI")
    # docs = [Document("I am a student"),Document("who to go to china"),Document("this is a table")]
    # vectorestore = langchainLib.create_faiss_from_docs(docs,embedding=embed)
    # langchainLib.save_faiss("faiss.db",vectorestore,index_name="gemini")

async def __loader_test(langchainLib:LangchainLib):
    result = langchainLib.load_html_split_markdown(url = "https://python.langchain.com/v0.2/docs")
    print("result:",[{"doc_len":len(doc['doc'].page_content),"doc_blocks":len(doc['blocks'])} for doc in result])
    blocks = []
    for item in result:
        blocks.extend(item['blocks'])
    print(len(blocks))
    vectorestore = langchainLib.create_faiss_from_docs(blocks)
    langchainLib.save_faiss("faiss.db",vectorestore,index_name="langchain_doc")

def __tools_test(langchainLib:LangchainLib,args):
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
    llm_key = args.llm
    llm = langchainLib.get_llm(llm_key)
    print("\n\nllms:",[(item["type"],item["api_key"],item["used"]) for item in langchainLib.llms],"\n")

    outputParser = langchainLib.get_outputParser(Output,fix=True,llm = llm,retry=3)
    prompt = langchainLib.get_prompt(human_keys = {"ask":"问题"},is_chat=False)
    search = langchainLib.get_search_tool("TAVILY")
    tools = [search]
    #chain  = prompt | (lambda x: x.messages[-1].content) | search | (lambda x:'```json\n{"name":中国,"age":"20"}') | outputParser
    #chain  = prompt | llm | outputParser
    chain = prompt | RunnableLambda(lambda x:x.text,name="抽取text") | search 
    #print("chain to_json = ",chain.to_json())
    #print("chain name=",chain.get_name())
    print("chain graph=",chain.get_graph(),"\n")
    res = chain.invoke({"ask": "北京人口多少"})
    print(type(res),res)
    FileLib.writeFile("graph1.png",chain.get_graph().draw_mermaid_png(),mode="wb")

    # llm = langchainLib.get_llm("LLM.DEEPSEEK")
    # print("llm",llm.to_json())
    #res = llm.invoke("你不擅长计算问题，遇到计算问题交给tool来完成")
    #print(res)
def __graph_test(langchainLib:LangchainLib):
    def multiply(one: int, two:int):
        """Multiply two numbers"""
        return one * two
    llm = langchainLib.get_llm()
    llm_with_tools = llm.bind_tools([multiply])
    graph = MessageGraph()
    graph.add_node("oracle",llm_with_tools)
    tool_node = ToolNode([multiply])
    graph.add_node("multiply",tool_node)
    graph.add_edge(START,"oracle")
    graph.add_edge("multiply",END)
    def router(state)-> Literal["multiply","__end__"]:
        tool_calls = state[-1].additional_kwargs.get("tool_nodes",[])
        if len(tool_calls):
            return "multiply"
        else:
            return END
    graph.add_conditional_edges("oracle",router)
    chain = graph.compile()
    FileLib.writeFile("graph.png",chain.get_graph(xray=True).draw_mermaid_png(),mode="wb")

    res = chain.invoke("202*308是多少")
    print(res)

async def start(args):
    langchainLib = LangchainLib()
    if args.mode == "llm":
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试llm {Color.RESET}")
        __llm_test(langchainLib,args)
    elif args.mode == "prompt":
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试prompt {Color.RESET}")
        await __prompt_test(langchainLib)    
    elif args.mode == 'loader':    
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试loader {Color.RESET}")
        await __loader_test(langchainLib)    
    elif args.mode == 'runnable':
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试runnable {Color.RESET}")
        __runnalble_test(langchainLib)
    elif args.mode == 'outputParser':    
        StringLib.logging_in_box(f"{Color.YELLOW} 测试outputParser {Color.RESET}")
        __outputParser_test(langchainLib)
    elif args.mode == 'rag':    
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试rag {Color.RESET}")
        __rag_test(langchainLib)
    elif args.mode == 'tools':
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试tools {Color.RESET}")
        __tools_test(langchainLib,args)
    elif args.mode == 'graph':
        StringLib.logging_in_box(f"\n{Color.YELLOW} 测试graph {Color.RESET}")
        __graph_test(langchainLib)
    else:
        __other(langchainLib,args)
    