from typing import Optional,Union,Literal
from ylz_utils.config import Config
import logging

from neo4j import GraphDatabase, Result
from neo4j.graph import Node,Relationship

from ylz_utils.data import StringLib

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
    
    def query(self,query:str,only_data=False,**kwargs):
        if not self.driver:
            self.get_driver()
        try:
            records,summary,keys = self.driver.execute_query(query,database_="neo4j",**kwargs) 
            if only_data :
                return [ record.data() for record in records]
            else:
                return records,summary,keys   
        except Exception as e:
            logging.error(e)
            raise e
    def create_relationships(self,*,objects=None,subjects=None,relationships,object_label,subject_label=None,key:str="name"):
        if not object_label:
            raise Exception("必须定义object_label!")
        if not subject_label:
            subject_label = object_label
            print(f"when subject_label is None,then subject_label=object_label={{{object_label}}}")
        '''
        创建主体节点(objects)和客体节点(subjects),主体和客体节点可以重复,以key为唯一标识。
          - 如果objects存在则merge方式创建objects
          - 如果subjects存在则merge方式创建subjects
          - objects与subjects的第一行属性作为节点属性,每一行都应保持同样的结构
          - object_label必须指定
          - 如果subject_label未指定则默认与object_label相同
        创建关系根据relationships,为(object)-[type]->(subject)关系表
          - 如果relationships存在则merge方式创建relation
          - 每一行结构必须包含object,subject,type字段，分别代表主体，客体和关系
          - 第一行属性(除object,subject,type外)作为关系属性，相同type的每一行都应保持同样的结构
          
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
        with self.driver.session() as session:
            # 批量创建节点和关系
            if objects:
                object_properties = "{" + ",".join([f"{key}:nodea.{key}" for key in objects[0]]) + "}"
                create_objects = f"""
                UNWIND $objects AS nodea
                MERGE (n:{object_label} {object_properties})
                """
                print(StringLib.lred(create_objects))
                session.run(create_objects, objects=objects)
            if subjects:            
                subject_properties = "{" + ",".join([f"{key}:nodeb.{key}" for key in subjects[0]]) + "}"
                create_subjects = f"""
                UNWIND $subjects AS nodeb
                MERGE (m:{subject_label} {subject_properties})
                """
                print(StringLib.green(create_subjects))
                session.run(create_subjects,subjects=subjects)
            
            if relationships:
                group_relation_types = set([ item["type"] for item in relationships ])
                group_relations =[]
                for relation_type in group_relation_types:
                    group_relations.append([item for item in relationships if item["type"]==relation_type])
                
                for relations in group_relations:
                    relation_type = relations[0]["type"]
                    relation_prop = [ key for key in relations[0].keys() if key not in ['object','subject','type']]
                    relation_properties = "{" + ",".join([f"{key}:rel.{key}" for key in relation_prop]) + "}"
                    
                    create_relationships = f"""
                      UNWIND $relations AS rel
                      MATCH (a:{object_label} {{{key}: rel.object}}), (b:{subject_label} {{{key}: rel.subject}})
                      MERGE (a)-[:{relation_type} {relation_properties}]->(b)
                    """
                    print(StringLib.yellow(create_relationships))
                    session.run(create_relationships,relations=relations)