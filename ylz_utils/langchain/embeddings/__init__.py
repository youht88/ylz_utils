import random
import time
from typing import Optional

from ylz_utils import Config

from langchain_together import TogetherEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_huggingface import HuggingFaceEndpointEmbeddings,HuggingFaceEmbeddings

from langchain_core.embeddings import DeterministicFakeEmbedding,Embeddings

class EmbeddingLib():
    embeddings:list  = []
    default_embedding_key = None
    fake_size:int = 0
    def __init__(self,langchainLib):
        self.langchainLib = langchainLib
        self.config = Config.get()
        self.regist_embedding()
        if self.config.get("EMBEDDING.DEFAULT"):
            default_embedding_key = self.config.get("EMBEDDING.DEFAULT")
            self.set_default_embedding_key(default_embedding_key)
        else:
            self.clear_default_embedding_key()
    def set_default_embedding_key(self,key):
        embeddings = [item for item in self.embeddings if item['type']==key]
        if embeddings:
            self.default_embedding_key = key
        else:
            self.clear_default_embedding_key()
    def get_default_embedding_key(self):
        return self.default_embedding_key
    def clear_default_embedding_key(self):
        self.default_embedding_key=None
    def set_fake_size(self,size=0):
        if size:
            self.fake_size = size
        else:
            self.fake_size = 0
    def get_embedding(self,key=None,model=None, full=False, fake_size: Optional[int]=None) :
        if full:
            return self.embeddings
        if fake_size:
            return DeterministicFakeEmbedding(size = fake_size)
        if self.fake_size:
            return DeterministicFakeEmbedding(size = self.fake_size)
        if self.embeddings:
            if not key:
                if self.default_embedding_key:
                    embeddings = [item for item in self.embeddings if item['type']==self.default_embedding_key]
                else:
                    embeddings = self.embeddings
            else:
                embeddings = [item for item in self.embeddings if item['type']==key]
            if embeddings:
                #embedding = random.choice(embeddings)    
                if len(embeddings)!=1:
                    raise Exception("基于安全的考虑，必须指定embedding的key! 比如key=EMBEDDING.HF")
                embedding = embeddings[0]
                if not embedding.get('embedding'):
                    embed_type = embedding['type']
                    if  embed_type == 'EMBEDDING.TOGETHER':
                        embedding['embedding'] = TogetherEmbeddings(api_key = embedding.get('api_key'),model=embedding.get('model'))
                    elif embed_type == "EMBEDDING.GEMINI" :
                        embedding['embedding'] = GoogleGenerativeAIEmbeddings(model=embedding.get('model'),google_api_key=embedding.get('api_key'))
                    elif embed_type == 'EMBEDDING.OLLAMA':
                        embedding['embedding'] = OllamaEmbeddings(model=embedding.get('model'))
                    elif embed_type == 'EMBEDDING.HF':
                        embedding['embedding'] = HuggingFaceEmbeddings(model_name=embedding.get('model'),
                                                                       model_kwargs = {'device': 'cpu'},
                                                                       encode_kwargs = {'normalize_embeddings': False})
                    else:
                        raise Exception(f"目前不支持{embedding['type']}嵌入模型")
                embedding['used'] = embedding.get('used',0) + 1 
                embedding['last_used'] = time.time()
                return embedding['embedding']
        if self.default_embedding_key:
            raise Exception(f"请确保{self.default_embedding_key}_API_KEYS环境变量被正确设置,然后调用regist_embedding注册语言模型")
        else:
            raise Exception(f"请确保<EMBEDDING PROVIDER>_API_KEYS环境变量被正确设置,然后调用regist_embedding注册语言模型")
    
    def regist_embedding(self):
        defaults = {
                      "EMBEDDING.TOGETHER": {"model":"BAAI/bge-large-en-v1.5"},
                      "EMBEDDING.GEMINI": {"model":"models/embedding-001"},
                      "EMBEDDING.OLLAMA": {"model":"mxbai-embed-large"},
                      "EMBEDDING.HF": {"model":"BAAI/bge-large-en"}  #"sentence-transformers/all-mpnet-base-v2"
                  }
        for key in defaults:
            default = defaults[key]
            embed = self.config.get(key)
            if not embed:
                continue
            base_url = embed.get("BASE_URL")
            api_keys = embed.get("API_KEYS")
            api_keys = self.langchainLib.split_keys(api_keys)

            model= embed.get("MODEL") if embed.get("MODEL") else default['model']
            for api_key in api_keys:
                self.embeddings.append({
                    "embedding": None,
                    "type": key,
                    "base_url": base_url,
                    "api_key":api_key,
                    "model":model,
                    "used":0,
                    "last_used": 0 
                })
    
    def embed_documents(self,embed:Embeddings,texts:list[str]):
        return embed.embed_documents(texts)
    
    def embed_query(self,embed:Embeddings,text:str):
        return embed.embed_query(text)
    
