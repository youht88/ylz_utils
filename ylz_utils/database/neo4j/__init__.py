from typing import Optional,Union,Literal
from ylz_utils.config import Config
import logging

from neo4j import GraphDatabase
from neo4j.graph import Node,Relationship

class Neo4jLib():
    def __init__(self,host:str=None,user:str=None ,password:str=None):
        self.config = Config()
        self.host = host or self.config.get("NEO4J.HOST")
        self.user = user or self.config.get("NEO4J.USER")
        self.password = password or self.config.get("NEO4J.PASSWORD")
        self.get_driver()

    def get_driver(self):
        self.driver = GraphDatabase.driver(self.host,auth=(self.user,self.password))
        return self.driver
    
    def close(self):
        if self.driver:
            self.driver.close()

    def run(self,command:str,**kwargs):
        if not self.driver:
            self.driver = self.get_driver()
        with self.driver.session(database="neo4j") as session:
            try:
                return session.run(command,**kwargs)
            except Exception as e:
                logging.error(e)
                raise e
    
    def query(self,query:str,**kwargs):
        if not self.driver:
            self.get_driver()
        try:
            return self.driver.execute_query(query,database_="neo4j",**kwargs)    
        except Exception as e:
            logging.error(e)
            raise e