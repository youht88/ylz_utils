from ylz_utils.langchain import LangchainLib
from ylz_utils.data import Spinner, StringLib

from langchain_core.messages import HumanMessage

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
    images=[]
    while True:
        if not message:
            message=input_with_readline(f"{user_id}用户: ")
        else:
            print(f"USER: {message}")
        message = message.strip()
        if message.lower() == '/h':
            print(StringLib.lyellow("/q:"),"退出",end=',')
            print(StringLib.lyellow("/c:"),"清空记录",end=',')
            print(StringLib.lyellow("/u <userid>:"),"设置用户id",end=',')
            print(StringLib.lyellow("/m <image_url>:"),"增加图片",end=',')
            print(StringLib.lyellow("/flux <描述想要生成的图像>:"),"生成图片")
            message = ""
            continue
        if message.lower() in ['/quit','/exit','/stop','/bye','/q']:
            print("Good bye!!")
            break
        if message.lower() == '/c':
            prompt = langchainLib.get_prompt(use_chat=True)
            message = ""
            continue
        if message.lower().startswith('/u'):
            new_user_id = message.split(" ")[1]
            if new_user_id:
                user_id = new_user_id
            message=""
            continue
        if message.lower().startswith('/m'):
            image_url = message.split(" ")[1]
            if image_url:
                images.append({"image":image_url})
            message = ""
            continue
        if message.lower().startswith('/flux'):
            image_message = message.split(" ")[1]
            if image_message:
                spinner = Spinner()
                spinner.start()
                file_name = langchainLib.fluxLib.gen(image_message)
                response = f"已生成图像{file_name},请查看"
                spinner.end()
                print(StringLib.color(response,["blue"]))
            message = ""
            continue
        if images:
            humanMessage = HumanMessage(content = images + [{"text":message}])
            res = llm.stream([humanMessage],{'configurable': {'user_id': user_id,'conversation_id': conversation_id}})
            print("AI: ",end="")
            try:
                for chunk in res:
                    print(chunk.content[0]['text'],end="")
            except:
                print("error!!!!")
                print(type(chunk))
                print(chunk)
            images = []
            message = ""
            print("\n")        
        else:
            res = chain.stream({"input":message}, {'configurable': {'user_id': user_id,'conversation_id': conversation_id}} )
            images = []
            message = ""
            print("AI: ",end="")
            for chunk in res:
                print(chunk,end="")
            print("\n")
    print("\n",langchainLib.get_llm(full=True))