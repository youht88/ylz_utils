from ylz_utils.config import Config
from ylz_utils.data import StringLib

import time
import random

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import SQLChatMessageHistory,ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

from gradio_client import Client,file

class LLMLib():
    llms:list = [] 
    default_llm_key = None
    chat_dbname = None
    def __init__(self):
        self.config = Config.get()
        self.regist_llm()
        if self.config.get("LLM.DEFAULT"):
            default_llm_key = self.config.get("LLM.DEFAULT")
            self.set_default_llm_key(default_llm_key)
        else:
            self.clear_default_llm_key()
        #self.add_plugins()
    def set_default_llm_key(self,key):
        llms = [item for item in self.llms if item['type']==key]
        if llms:
            self.default_llm_key = key
        else:
            self.clear_default_llm_key()
    def get_default_llm_key(self):
        return self.default_llm_key
    def clear_default_llm_key(self):
        self.default_llm_key=None
    def set_dbname(self,dbname):
        self.chat_dbname = dbname
    def get_user_session_history(self, user_id: str, conversation_id: str):
        if self.chat_dbname:
            return SQLChatMessageHistory(f"{user_id}--{conversation_id}", f"sqlite:///{self.chat_dbname}")
        raise Exception("you must give the chat_dbname . please use set_dbname first!")
        
    def get_chat(self,llm,prompt,history_messages_key = "history",input_key = "input"):
        return RunnableWithMessageHistory(
            prompt | llm,
            self.get_user_session_history,
            input_messages_key = input_key,
            history_messages_key= history_messages_key,
            history_factory_config=[
                    ConfigurableFieldSpec(
                        id="user_id",
                        annotation=str,
                        name="User ID",
                        description="Unique identifier for the user.",
                        default="",
                        is_shared=True,
                    ),
                    ConfigurableFieldSpec(
                        id="conversation_id",
                        annotation=str,
                        name="Conversation ID",
                        description="Unique identifier for the conversation.",
                        default="",
                        is_shared=True,
                    ),
                ]
        )

    def get_llm(self,key=None, model=None, temperature=None, full=False, delay=10)->ChatOpenAI | ChatOllama:
        if full:
            return self.llms
        if self.llms:
            if not key:
                if self.default_llm_key:
                    llms = [item for item in self.llms if item['type']==self.default_llm_key]
                else:
                    llms = self.llms
            else:
                if key == "LLM.GEMINI":
                    #macos不支持
                    llms=[]
                else:
                    llms = [item for item in self.llms if item['type']==key]
            if llms:
                while True:
                    llm = random.choice(llms) 
                    #print(llm.get('last_used',0),time.time() - llm.get('last_used'),llm['api_key'])   
                    over_delay = (time.time() - llm.get('last_used',0)) > delay # 3秒内不重复使用
                    if not over_delay:
                        continue
                    if not model and llm['llm']==None:
                        llm['model'] = llm['default_model']
                    elif model and model != llm.get('model'):
                        llm['model'] = model
                        llm['llm'] = None 
                    if not llm.get('llm'):
                        llm_type = llm.get('type')
                        if llm_type == 'LLM.GEMINI':
                            try:
                                StringLib.logging_in_box(f"google_api_key={llm.get('api_key')},model={llm.get('model')}")
                                llm['llm'] = ChatGoogleGenerativeAI(
                                    google_api_key=llm.get('api_key'), 
                                    model=llm.get('model'),
                                    temperature= temperature or llm.get('temperature'))
                            except:
                                raise Exception(f"请确保{llm_type}_API_KEYS环境变量被正确设置")                                
                        elif llm_type == 'LLM.QIANFAN':
                            try:
                                StringLib.logging_in_box(f"qianfan_api_key={llm.get('api_key')},qianfan_sec_key={llm.get('sec_key')},model={llm.get('model')}")
                                llm['llm'] = QianfanChatEndpoint(
                                    qianfan_ak= llm.get('api_key'),
                                    qianfan_sk= llm.get('sec_key'),
                                    model= llm.get('model'),
                                    temperature = temperature or llm.get('temperature'))
                            except:
                                raise Exception(f"请确保{llm_type}_API_KEYS和{llm_type}_SEC_KEYS环境变量被正确设置")
                        elif llm_type == 'LLM.OLLAMA':
                            try:
                                StringLib.logging_in_box(f"ollama_api_key={llm.get('api_key')},model={llm.get('model')}")
                                llm['llm'] = ChatOllama(model= llm.get('model'),
                                                        temperature= temperature or llm.get('temperature'),
                                                        keep_alive=llm.get('keep_alive'))
                            except:
                                raise Exception(f"请确保{llm_type}_API_KEYS环境变量被正确设置")
                            
                        else:
                            try:
                                llm['llm'] = ChatOpenAI(
                                    base_url=llm.get('base_url'),
                                    api_key= llm.get('api_key'),
                                    model= llm.get('model'),
                                    temperature= temperature or llm.get('temperature')
                                    )
                            except:
                                raise Exception(f"请确保{llm_type}_API_KEYS环境变量被正确设置")                                
                            
                    llm['used'] = llm.get('used',0) + 1 
                    llm['last_used'] = time.time()
                    return llm['llm'] 
        if self.default_llm_key:
            raise Exception(f"请确保{self.default_llm_key}_API_KEYS环境变量被正确设置,然后调用regist_llm注册语言模型")
        else:
            raise Exception(f"请确保<LLM PROVIDER>_API_KEYS环境变量被正确设置,然后调用regist_llm注册语言模型")        
    def regist_llm(self):
        defaults = {"LLM.TOGETHER":
                      {"model":"Qwen/Qwen1.5-72B-Chat","temperature":0},
                   "LLM.SILICONFLOW":
                      {"model":"alibaba/Qwen1.5-110B-Chat","temperature":0},
                   "LLM.GROQ":
                      {"model":"llama-3.1-70b-versatile","temperature":0},
                   "LLM.GEMINI":
                      {"model":"gemini-pro","temperature":0},
                    "LLM.DEEPSEEK":
                      {"model":"deepseek-chat","temperature":1},
                    "LLM.QIANFAN":
                      {"model":"Yi-34B-Chat","temperature":0.7},
                    "LLM.OLLAMA":
                      {"model":"llama3.1","temperature":0.7},
                    "LLM.MOONSHOT":
                      {"model":"moonshot-v1-8k","temperature":0.3}
                  }
        for key in defaults:
            default = defaults[key]
            llm = self.config.get(key)
            if not llm:
                continue
            base_url = llm.get("BASE_URL")
            api_keys = llm.get("API_KEYS")
            keep_alive = llm.get("KEEP_ALIVE")
            if api_keys:
                api_keys = api_keys.split(",")
                if not api_keys[-1]:
                    api_keys.pop()
            else:
                api_keys = []

            sec_keys = llm.get("SEC_KEYS")
            if sec_keys:
                sec_keys = sec_keys.split(",")
                # 防止最后
                if not sec_keys[-1]:
                    sec_keys.pop()
            else:
                sec_keys = []
            model= llm.get("MODEL") if llm.get("MODEL") else default['model']
            temperature = llm.get("TEMPERATURE") if llm.get("TEMPERATURE") else default['temperature']
            for idx, api_key in enumerate(api_keys):
                if key == "LLM.QIANFAN" and len(api_keys) == len(sec_keys):
                    sec_key = sec_keys[idx]
                else:
                    sec_key = ""
                self.llms.append({
                    "llm": None,
                    "type": key,
                    "base_url":base_url,
                    "api_key":api_key,
                    "sec_key": sec_key,
                    "default_model": model,
                    "model":model,
                    "temperature":temperature,
                    "keep_alive":keep_alive,
                    "used":0,
                    "last_used": 0 
                })
    def get_opengpt4o_client(self,api_key):
        client = Client("KingNish/OpenGPT-4o")
        return client
    def opengpt4o_predict(self, client,prompt="hello",imageUrl=None):
        imageFile = file(imageUrl) if imageUrl else None
        result = client.predict(
                image3=imageFile,
                prompt3=prompt,
                api_name="/predict"
        )
        return result
    def opengpt4o_chat(self,client,prompt="hello",imageUrls=[],temperature=0.5,webSearch=True):
        imageFiles = [file(imageUrl) for imageUrl in imageUrls] if imageUrls else []
        result = client.predict(
            message={"text":prompt,"files":imageFiles},
            request="idefics2-8b-chatty",
            param_3="Top P Sampling",
            param_4=temperature,
            param_5=4096,
            param_6=1,
            param_7=0.9,
            param_8=webSearch,
            api_name="/chat"
        )
        return result
    def get_chatopenai_llm(self,base_url,api_key,model,temperature=0.7):
        llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature
                )
        return llm
