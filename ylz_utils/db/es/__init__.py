from datetime import datetime
from elasticsearch_dsl import connections,Document,Date,Integer,Keyword,Text,Long,Double,Float,Boolean
from elasticsearch_dsl import Search,Q
from elasticsearch_dsl.query import MultiMatch
from typing import Optional,Union,Literal

class ESLib():
    db = None
    analyzer='ik_max_word'
    search_analyzer='ik_smart'
    indexs = {}
    def __init__(self,hosts:list[str]=['https://localhost:9200'],
                 user_passwd:tuple[str,str]=('elastic','abcd1234'),
                 alias:str='es',
                 verify_certs=False,
                 ssl_show_warn=False,
                 analyzer='ik_max_word',search_analyzer='ik_smart'):
        self.db = alias
        self.analyzer = analyzer
        self.search_analyzer = search_analyzer
        connections.create_connection(
            hosts=hosts,
            basic_auth=user_passwd,
            alias=alias,
            verify_certs=verify_certs,
            ssl_show_warn=ssl_show_warn
        )
        self.Q = Q
        self.Search = Search
        self.MultiMatch = MultiMatch
    
    def register_class(self,cls:Document):
        setattr(self, cls.__name__, cls)  
    def register(self, class_name, index_name, shards:int=None,replicas:int=None,
                 fields:dict[str,Literal['integer','long','float','double','boolean','ktext','text','keyword','date']]={}):
        """动态创建Document类。

        Args:
            class_name (str): 类名。
            index_name (str): index的名称
            shards (int): shards数量
            replicas (int): replicas数量
            field (dict, optional): 字段字典。 默认为空字典。
        Returns:
            type: 新创建Document类。
        """
        base_classes=(Document,)
        index_fields = {}
        for key in fields:
            field_type = fields[key]
            if field_type=='integer':
                index_fields[key] = Integer()
            elif field_type=='long':
                index_fields[key] = Long()
            elif field_type=='float':
                index_fields[key] = Float()
            elif field_type=='double':
                index_fields[key] = Double()
            elif field_type=='boolean':
                index_fields[key] = Boolean()
            elif field_type=='ktext':
                index_fields[key] = Text(analyzer=self.analyzer,search_analyzer=self.search_analyzer,fields={'keyword':Keyword()})
            elif field_type=='text':
                index_fields[key] = Text(analyzer=self.analyzer,search_analyzer=self.search_analyzer)
            elif field_type=='keyword':
                index_fields[key] = Keyword()
            elif field_type=='date':
                index_fields[key] = Date()
            else:
                raise Exception(f"ES不支持的数据类型:{field_type}")                

        # 创建 Index 类
        settings={}
        if shards:
            settings["number_of_shards"] = shards
        if replicas:
            settings["number_of_replicas"] = replicas
        Index = type('Index', tuple(), {'name': index_name, 'settings': settings})
        index_fields['Index'] = Index
        cls = type(class_name, base_classes, index_fields)        
        setattr(self, cls.__name__, cls)
        self.indexs[index_name] = cls
        return cls
    def matchQ(self,column_name,value):
        return Q({"match":{column_name:{"query":value,"analyzer":"ik_smart"}}})
    def termQ(self,column_name,value):
        return Q({"term":{column_name + ".keyword":value}})
    def multimatchQ(self,column_names,value):
        return MultiMatch(query=value,fields=column_names,analyzer=self.search_analyzer)
    def search(self,index_name,q,start=0,end=9):
        s = Search(using = self.db,index=index_name).query(q)
        res = s[start:end].execute()
        cls = self.indexs[index_name]
        #print(res[0].to_dict())
        results = [cls(**data.to_dict()) for data in res]
        size = res.hits.total.value
        return results,size
if __name__ == '__main__':
    esLib = ESLib(user_passwd=('elastic','9HIMozq48xIP+PHTpRVP'))
    # class Product(Document):
    #     id = Integer()
    #     name = Text(analyzer='ik_max_word',search_analyzer='ik_smart',fields={'keyword':Keyword()})
    #     summary = Text(analyzer='ik_max_word',search_analyzer='ik_smart')
    #     solution_class = Text(analyzer='ik_max_word',search_analyzer='ik_smart')
    #     tags = Keyword()
    #     published_from = Date()
    #     class Index():
    #         name = 'product'
    #         settings = {
    #             "number_of_shards": 1,
    #             "number_of_replicas": 0,
    #         }
    # esLib.register_class(Product)
    # esLib.Product.init(using="es")
    # product = esLib.Product.get(id=42,using='es')
    # print(product)
    Product = esLib.register("Product","product",fields={
        "id": "integer",
        "name":"ktext",
        "summary":"text",
        "tags":"keyword",
        "published_from":"date"
    })
    Product.init(using="es")    
    # product = Product(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags='Linabell')
    # product.publish_from = datetime.now()
    # product.save(using="es")
    q = esLib.termQ("tags","Linabell")
    res,size = esLib.search("product",q) 
    if res:
        print(res[0].name,size)
    else:
        print("no result")
    
