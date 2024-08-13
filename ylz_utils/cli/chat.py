from ylz_utils.langchain import LangchainLib

def input_with_readline(prompt):
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print("输入的内容存在编码问题，请确保使用 UTF-8 编码的字符，并不要使用回退键。")
        return input_with_readline(prompt)
    
def start_chat(langchainLib:LangchainLib,args):
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
            message=input_with_readline("USER: ")
        else:
            print(f"USER: {message}")
        if message.lower() in ['/quit','/exit','/stop','/bye','/q']:
            print("Good bye!!")
            break
        res = chain.stream({"input":message}, {'configurable': {'user_id': user_id,'conversation_id': conversation_id}} )
        message = ""
        print("AI: ",end="")
        for chunk in res:
            print(chunk,end="")
        print("\n")
    print("\n",langchainLib.get_llm(full=True))