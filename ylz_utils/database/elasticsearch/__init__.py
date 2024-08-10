from datetime import datetime
from elasticsearch_dsl import Index as dsl_Index
from elasticsearch_dsl import connections, Document, Date, Nested, Boolean, \
    analyzer, InnerDoc, Completion, Keyword, Text,Integer,Long,Double,Float,\
    DateRange,IntegerRange,FloatRange,IpRange,Ip,Range
from elasticsearch_dsl import Search,Q
from elasticsearch_dsl.query import MultiMatch
from typing import Optional,Union,Literal

class ESLib():
    using = None
    analyzer='ik_max_word'
    search_analyzer='ik_smart'
    indexes = {}
    def __init__(self,hosts:list[str]=['https://localhost:9200'],
                 user_passwd:tuple[str,str]=('elastic','abcd1234'),
                 using:str = "es",
                 verify_certs=False,
                 ssl_show_warn=False,
                 analyzer='ik_max_word',search_analyzer='ik_smart'):
        self.using = using
        self.analyzer = analyzer
        self.search_analyzer = search_analyzer
        connections.create_connection(
            hosts=hosts,
            basic_auth=user_passwd,
            alias=using,
            verify_certs=verify_certs,
            ssl_show_warn=ssl_show_warn
        )
        self.Q = Q
        self.Search = Search
        self.MultiMatch = MultiMatch
    
    def register_class(self,cls:Document):
        setattr(self, cls.__name__, cls) 
        cls.init(using=self.using) 
        return cls
    def register(self, class_name, index_name, shards:int=None,replicas:int=None,
                 fields:dict[str,Literal['int','long','float',\
                                         'double','boolean','ktext',\
                                         'text','keyword','datetime','date','ip' \
                                         'int_range','float_range','date_range','ip_range'
                                         ]]={}):
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
            field_type = fields[key].get("type")
            multi = fields[key].get("multi")
            required = fields[key].get("requried")
            if field_type=='int':
                index_fields[key] = Integer(multi=multi,required=required)
            elif field_type=='long':
                index_fields[key] = Long(multi=multi,required=required)
            elif field_type=='float':
                index_fields[key] = Float(multi=multi,required=required)
            elif field_type=='double':
                index_fields[key] = Double(multi=multi,required=required)
            elif field_type=='boolean':
                index_fields[key] = Boolean(multi=multi,required=required)
            elif field_type=='ktext':
                index_fields[key] = Text(analyzer=self.analyzer,search_analyzer=self.search_analyzer,fields={'keyword':Keyword()},multi=multi,required=required)
            elif field_type=='text':
                index_fields[key] = Text(analyzer=self.analyzer,search_analyzer=self.search_analyzer,multi=multi,required=required)
            elif field_type=='keyword':
                index_fields[key] = Keyword(multi=multi,required=required)
            elif field_type=='datetime':
                index_fields[key] = Date(multi=multi,required=required)
            elif field_type=='date':
                index_fields[key] = Date(format="yyyy-MM-dd", multi=multi,required=required)
            elif field_type=='ip':
                index_fields[key] = Ip(multi=multi,required=required)
            elif field_type=='date_range':
                index_fields[key] = DateRange(multi=multi,required=required)
            elif field_type=='ip_range':
                index_fields[key] = IpRange(multi=multi,required=required)
            elif field_type=='int_range':
                index_fields[key] = IntegerRange(multi=multi,required=required,default=default)
            elif field_type=='float_range':
                index_fields[key] = FloatRange(multi=multi,required=required,default=default)
            else:
                raise Exception(f"ESLib不支持的数据类型:{field_type}")                

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
        self.indexes[index_name] = cls
        cls.init(using=self.using)
        return cls
    def matchQ(self,column_name,value):
        return Q({"match":{column_name:{"query":value,"analyzer":"ik_smart"}}})
    def termQ(self,column_name,value):
        return Q({"term":{column_name + ".keyword":value}})
    def multimatchQ(self,column_names,value):
        return MultiMatch(query=value,fields=column_names,analyzer=self.search_analyzer)
    def search(self,index_name,q=None,start=0,end=9):
        if q:
            s = Search(using = self.using,index=index_name).query(q)
            res = s[start:end].execute()
            cls = self.indexes[index_name]
            results = [cls(**data.to_dict()) for data in res]
            #size = res.hits.total.value
            return results
        else:
            cls = self.indexes[index_name]
            s = cls.search(using=self.using)
            res = s[start:end].execute()
            return res.hits
    def get(self,index_name,id:str,missing:Literal["none","raise","skip"]="none"):
        cls = self.indexes[index_name]
        doc = cls.get(using=self.using,id=id,missing=missing)
        return doc
    def mget(self,index_name,ids:list[str],missing:Literal["none","raise","skip"]="none"):
        cls = self.indexes[index_name]
        doc = cls.mget(using=self.using,docs=ids,missing=missing)
        return doc
    def index_exists(self,index_name) -> bool:
        index = dsl_Index(index_name)
        return index.exists_alias(using=self.using,name=index_name)
    # def index_create(self,index_name) -> bool:
    #     cls = self.indexes[index_name]
    #     return cls.save(using=self.using)
    def index_delete(self,index_name) -> bool:
        index = dsl_Index(index_name)
        return index.delete(using=self.using)
    def doc_dict(self,doc):
        return doc.to_dict()
    def doc_save(self,doc):
        return doc.save(using=self.using)
    def doc_update(self,doc,**kwargs):
        return doc.update(using=self.using,**kwargs)
    def doc_delete(self,doc):
        return doc.delete(using=self.using)
        
if __name__ == '__main__':
    esLib = ESLib(user_passwd=('elastic','9HIMozq48xIP+PHTpRVP'),using='es')
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
    # Product.init(using="es")
    # product = Product()
    # product.save(using="es")
    #product = esLib.Product.get(id=42,using='es')
    #print(product)
    Product = esLib.register("Product","product_new",fields={
        "id": {"type":"int","required":True},
        "name":{"type":"ktext"},
        "summary":{"type":"text"},
        "tags":{"type":"keyword","multi":True},
        "published_from":{"type":"date"}
    })
    print("index_exists:",esLib.index_exists("product_new"))
    docs = esLib.search("product_new")
    product = Product(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags='Linabell')
    print("to_dict:",esLib.doc_dict(product))
    print("insert:",esLib.doc_save(product))
    print(esLib.mget("product_new",ids=[44,34]))
    print("update:",esLib.doc_update(product,summary="hello world"))
    print(esLib.mget("product_new",ids=[44,34]))
    print("delete:",esLib.doc_delete(product))
    print(esLib.mget("product_new",ids=[44,34]))
    print("index_delete",esLib.index_delete("product_new"))
    
    # for doc in docs:
    #     esLib.delete(doc)
    #product.save(using="es")
    #esLib.save("product_new")  
    # product = Product(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags='Linabell')
    # product.publish_from = datetime.now()
    # product.save(using="es")
    
    # q = esLib.termQ("tags","Linabell")
    # res,size = esLib.search("product",q) 
    # if res:
    #     print(res[0].name,size)
    # else:
    #     print("no result")
    
