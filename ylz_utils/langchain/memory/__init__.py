from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from langchain.memory import ConversationBufferMemory

from langchain_elasticsearch import ElasticsearchChatMessageHistory
from langchain_community.chat_message_histories import ElasticsearchChatMessageHistory
from langchain_elasticsearch.client import create_elasticsearch_client

from langchain_community.chat_message_histories import SQLChatMessageHistory,ChatMessageHistory

from ylz_utils.config import Config 

class MemoryLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = Config()
    def get_memory(self,dbname:str,user_id:str,conversation_id:str):
        if dbname:
            if dbname.startswith("es://"):
                # es://username:password@127.0.0.1:9200/index_name 
                # es:///index_name
                # sqlite:///db_name
                # db_name
                # :memory:
                if dbname.startswith("es:///"):
                    es_url = self.config.get("ES.HOST")
                    es_username = self.config.get("ES.USER")
                    es_password = self.config.get("ES.PASSWORD")
                    index_name = dbname.split(':///')[1]
                else:
                    es_url = f"https://{dbname.split('@')[1].split('/')[0]}"
                    es_username = dbname.split('@')[0].split('://')[1].split(":")[0]
                    es_password = dbname.split('@')[0].split('://')[1].split(":")[1]
                    index_name = dbname.split('@')[1].split('/')[1]
                session_id = f"{user_id}--{conversation_id}"
                #print(es_url,es_username,es_password,index_name,session_id)
                es_connection = create_elasticsearch_client(
                                url=es_url,
                                username=es_username,
                                password=es_password,
                                params = {"verify_certs":False,"ssl_show_warn":False},
                            )
                return  ElasticsearchChatMessageHistory(
                            es_connection=es_connection,
                            index=index_name,
                            session_id=session_id
                        )
            elif dbname.startswith("sqlite:///"):
                return SQLChatMessageHistory(f"{user_id}--{conversation_id}", f"{dbname}")
            elif dbname==":memory:":
                return SQLChatMessageHistory(f"{user_id}--{conversation_id}", ":memory:")
            else:
                return SQLChatMessageHistory(f"{user_id}--{conversation_id}", f"sqlite:///{dbname}")
        raise Exception("you must give the chat_dbname . please use set_dbname first!")
