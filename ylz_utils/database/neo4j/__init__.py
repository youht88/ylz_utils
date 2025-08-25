from typing import Optional,Union,Literal
from ylz_utils.config import Config
import logging
import re
from neo4j import GraphDatabase, Result,EagerResult,Record
from neo4j.graph import Node,Relationship

from ylz_utils.data import StringLib

class Neo4jLib():
    def __init__(self,host:str=None,user:str=None ,password:str=None,database:str='neo4j'):
        self.config = Config()
        self.host = host or self.config.get("NEO4J.HOST")
        self.user = user or self.config.get("NEO4J.USER")
        self.password = password or self.config.get("NEO4J.PASSWORD")
        self.database = database or self.config.get("NEO4J.DATABASE")
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
        with self.driver.session(database=self.database) as session:
            try:
                return session.run(command,**kwargs)
            except Exception as e:
                logging.error(e)
                raise e
        
    def query(self,query:str,**kwargs) -> EagerResult:
        if not self.driver:
            self.get_driver()
        try:
            result = self.driver.execute_query(query,database_=self.database,**kwargs) 
            return result   
        except Exception as e:
            logging.error(e)
            raise e
    
    def get_data(self,records:list[Record] | EagerResult):
        if isinstance(records,EagerResult):
            records = records.records
        return [ record.data() for record in records]
    
    def get_rel_schema(self):
        relType = self.get_data(self.query("Call db.schema.relTypeProperties()"))
        return relType
    def get_node_schema(self):
        nodeType = self.get_data(self.query("Call db.schema.nodeTypeProperties()"))
        return nodeType
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

    def create_node_and_relation(self,data:dict):
        with self.driver.session() as session:
            node_labels = [re.findall(r"(.*)_key",key)[0] for key in data.keys() if re.match(r".*_key",key)] 
            # merge nodes
            for node_label in node_labels:
                nodes = data.get(node_label)
                if nodes:
                    node_properties = "{" + ",".join([f"{key}:node.{key}" for key in nodes[0]]) + "}"
                    create_nodes = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{node_label} {node_properties})
                        """
                    print(StringLib.green(nodes))
                    session.run(create_nodes, nodes=nodes)
            # merge relation
            relations = data.get('relations',[])
            for relation in relations:
                node_from_value = relation.get("from")
                node_from_label = relation.get("from_label")
                node_from_key = relation.get("from_key")
                if not node_from_key:
                    node_from_key = data[f"{node_from_label}_key"]
                node_to_value = relation.get("to")
                node_to_label = relation.get("to_label")
                node_to_key = relation.get("to_key")
                if not node_to_key:
                    node_to_key = data[f"{node_to_label}_key"]
                relation_type = relation["type"]
                relation_prop = [ key for key in relation.keys() if key not in ["from","from_label","from_key","to","to_label","to_key"]]
                relation_properties = "{" + ",".join([f"{key}:rel.{key}" for key in relation_prop]) + "}"
                           
                if node_from_value and node_to_value and node_from_label and node_to_label and relation_type:
                    create_relations = f"""
                        WITH $relation as rel
                        MATCH (a:{node_from_label} {{{node_from_key}: "{node_from_value}"}}), (b:{node_to_label} {{{node_to_key}: "{node_to_value}"}})
                        MERGE (a)-[:{relation_type} {relation_properties}]->(b)
                    """
                    print(StringLib.green(relations))
                    session.run(create_relations,relation=relation)    
