from typing import Optional,Union,Literal
from ylz_utils.config import Config
import logging

from neo4j import GraphDatabase, Result
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

    def run(self,command:str,**kwargs) -> Result:
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
    def create_relationships(self,objects,subjects,relationships,*,object_label,subject_label,key:str="name"):
        '''
        创建主体节点(objects)和客体节点(subjects),主体和客体节点可以重复,以key为唯一标识。
        创建关系根据relationships,为(object)-[type]->(subject)关系表
        
        例如: 以下数据,key为name
        # 定义Person数据
        objects = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
        subjects = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]

        # 定义关系数据
        relationships = [
            {"object": "Alice", "subject": "Bob","type":"朋友"},
            {"object": "Alice", "subject": "Charlie","type":"领导"},
            {"object": "Bob", "subject": "Charlie","type":"同事"}
        ]
        create_relationships(objects,subjects,relationships,object_label='Person',subject_label='Person',key='name')
        '''
        object_properties = "{" + ",".join([f"{key}:nodea.{key}" for key in objects[0]]) + "}"
        subject_properties = "{" + ",".join([f"{key}:nodeb.{key}" for key in subjects[0]]) + "}"
        with self.driver.session() as session:
            # 批量创建节点和关系
            cypher_query = f"""
            UNWIND $objects AS nodea
            MERGE (n:{object_label} {object_properties})
            UNWIND $subjects AS nodeb
            MERGE (m:{subject_label} {subject_properties})

            WITH n
            UNWIND $relationships AS rel
            MATCH (a:{object_label} {{{key}: rel.from}}), (b:{subject_label} {{{key}: rel.to}})
            CREATE (a)-[:FRIEND]->(b)
            """

            # 执行查询
            session.run(cypher_query, objects=objects, subjects=subjects, relationships=relationships)