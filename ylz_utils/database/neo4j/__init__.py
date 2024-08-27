from typing import Optional,Union,Literal
from ylz_utils.config import Config

from neo4j import GraphDatabase
from neo4j.graph import Node,Relationship

class Neo4jLib():
    def __init__(self,host:str=None,user:str=None ,password:str=None):
        self.config = Config()
        self.host = host or self.config.get("NEO4J.HOST")
        self.user = user or self.config.get("NEO4J.USER")
        self.password = password or self.config.get("NEO4J.PASSWORD")
    def get_session(self):
        self.driver = GraphDatabase.driver(self.host,auth=(self.user,self.password))
        self.session = self.driver.session()
        return self.session
    def close_session(self):
        if self.session:
            self.session.close()
    def run(self,command:str):
        if not self.session:
            
            session = self.get_session()
        else:
            session = self.session
        return session.run(command)
        