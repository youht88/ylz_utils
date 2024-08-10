from datetime import datetime
from elasticsearch_dsl import Index as dsl_Index
from elasticsearch_dsl import connections, Field,Document, Date, Nested, Boolean, \
    analyzer, InnerDoc, Completion, Keyword, Text,Integer,Long,Double,Float,\
    DateRange,IntegerRange,FloatRange,IpRange,Ip,Range
from elasticsearch_dsl import Search,Q,FacetedSearch,MultiSearch
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
        self.MultiSearch = MultiSearch
        self.FacetedSearch = FacetedSearch
        self.MultiMatch = MultiMatch
    
    def register_class(self,cls:Document):
        setattr(self, cls.__name__, cls)
        self.indexes[cls.Index.name]=cls 
        cls.init(using=self.using) 
        return cls
    def _parse_fields(self,raw_fields):
        print("@@@@",raw_fields)
        wrapped_fields : dict[str,Field]= {}
        for key in raw_fields:
            field_type = raw_fields[key].get("type")
            multi = raw_fields[key].get("multi")
            required = raw_fields[key].get("requried")
            print("====",key,field_type)
            if field_type=='int':
                wrapped_fields[key] = Integer(multi=multi,required=required)
            elif field_type=='long':
                wrapped_fields[key] = Long(multi=multi,required=required)
            elif field_type=='float':
                wrapped_fields[key] = Float(multi=multi,required=required)
            elif field_type=='double':
                wrapped_fields[key] = Double(multi=multi,required=required)
            elif field_type=='boolean':
                wrapped_fields[key] = Boolean(multi=multi,required=required)
            elif field_type=='ktext':
                wrapped_fields[key] = Text(analyzer=self.analyzer,search_analyzer=self.search_analyzer,fields={'keyword':Keyword()},multi=multi,required=required)
            elif field_type=='text':
                wrapped_fields[key] = Text(analyzer=self.analyzer,search_analyzer=self.search_analyzer,multi=multi,required=required)
            elif field_type=='keyword':
                wrapped_fields[key] = Keyword(multi=multi,required=required)
            elif field_type=='datetime':
                wrapped_fields[key] = Date(multi=multi,required=required)
            elif field_type=='date':
                wrapped_fields[key] = Date(format="yyyy-MM-dd", multi=multi,required=required)
            elif field_type=='ip':
                wrapped_fields[key] = Ip(multi=multi,required=required)
            elif field_type=='date_range':
                wrapped_fields[key] = DateRange(multi=multi,required=required)
            elif field_type=='ip_range':
                wrapped_fields[key] = IpRange(multi=multi,required=required)
            elif field_type=='int_range':
                wrapped_fields[key] = IntegerRange(multi=multi,required=required)
            elif field_type=='float_range':
                wrapped_fields[key] = FloatRange(multi=multi,required=required)
            elif field_type=='innerdoc':
                innerdoc = raw_fields[key]["innerdoc"]
                innerdoc_name = innerdoc["name"]
                innerdoc_raw_fields = innerdoc["fields"]
                innerdoc_wrapped_fields = self._parse_fields(innerdoc_raw_fields)
                print("???",innerdoc_wrapped_fields)
                innerdoc_cls = type(innerdoc_name, (InnerDoc,), innerdoc_wrapped_fields)
                print("!!!!",innerdoc_cls(),isinstance(innerdoc_cls(),InnerDoc))
                setattr(self,innerdoc_cls.__name__,innerdoc_cls) 
                wrapped_fields[key] = innerdoc_cls()
            else:
                raise Exception(f"ESLib不支持的数据类型:{field_type}") 
        return wrapped_fields               

    def register(self, class_name, index_name, shards:int=None,replicas:int=None,
                 fields:dict[str,dict]={}):
        """动态创建Document类。

        Args:
            class_name (str): 类名。
            index_name (str): index的名称
            shards (int): shards数量
            replicas (int): replicas数量
            field (key:{"type":Literal['int','long','float',\
                                         'double','boolean','ktext',\
                                         'text','keyword','datetime','date','ip' \
                                         'int_range','float_range','date_range','ip_range'
                                         ],
                        "required":False,
                        "multi":False}): 字段字典。 默认为空字典。

        Returns:
            type: 新创建Document类。
        """
        base_classes=(Document,)
        wrapped_fields = self._parse_fields(fields)
        print("-----",wrapped_fields)
        # 创建 Index 类
        settings={}
        if shards:
            settings["number_of_shards"] = shards
        if replicas:
            settings["number_of_replicas"] = replicas
        Index = type('Index', tuple(), {'name': index_name, 'settings': settings})
        wrapped_fields['Index'] = Index
        cls = type(class_name, base_classes, wrapped_fields)        
        setattr(self, cls.__name__, cls)
        self.indexes[index_name] = cls
        cls.init(using=self.using)
        return cls
    def get_matchQ(self,column_name,value):
        return Q({"match":{column_name:{"query":value,"analyzer":"ik_smart"}}})
    def get_termQ(self,column_name,value):
        return Q({"term":{column_name + ".keyword":value}})
    def get_multimatchQ(self,column_names,value):
        return MultiMatch(query=value,fields=column_names,analyzer=self.search_analyzer)
    def index_search(self,index_name,q=None,start=0,end=9):
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
    def index_get(self,index_name,id:str,missing:Literal["none","raise","skip"]="none"):
        cls = self.indexes[index_name]
        doc = cls.get(using=self.using,id=id,missing=missing)
        return doc
    def index_mget(self,index_name,ids:list[str],missing:Literal["none","raise","skip"]="none"):
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
    class Product(Document):
        id = Integer()
        name = Text(analyzer='ik_max_word',search_analyzer='ik_smart',fields={'keyword':Keyword()})
        summary = Text(analyzer='ik_max_word',search_analyzer='ik_smart')
        solution_class = Text(analyzer='ik_max_word',search_analyzer='ik_smart')
        tags = Keyword()
        published_from = Date()
        class Index():
            name = 'product'
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
    esLib.register_class(Product)
    product0 = Product(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags=['Linabell'])
    esLib.doc_save(product0)
    products = esLib.index_search("product")
    print("class search:",[esLib.doc_dict(product) for product in products])    
    # product.save(using="es")
    #product = esLib.Product.get(id=42,using='es')
    #print(product)
    ProductNew = esLib.register("Product","product_new",fields={
        "id": {"type":"int","required":True},
        "name":{"type":"ktext"},
        "summary":{"type":"text"},
        "tags":{"type":"keyword","multi":True},
        "published_from":{"type":"date",},
        "info":{"type":"innerdoc","innerdoc":{"name":"Info","fields":{
            "provider":{"type":"text"},
            "price":{"type":"float"}
        }}}
    })
    print("index_exists:",esLib.index_exists("product_new"))
    docs = esLib.index_search("product_new")
    product1 = ProductNew(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags=['Linabell'])
    product1.info = esLib.Info(provider="abc",price=20.2)
    print("*"*20,product1.to_dict())
    product2 = ProductNew(meta={'id':34},id=1034,name='mouse',summary='a white mouse',solution_class='home',tags=['white','wireless'])
    product2.info = esLib.Info(provider="xyz",price=30.4)
    print("to_dict:",esLib.doc_dict(product1))
    print("insert:",esLib.doc_save(product1))
    print("insert:",esLib.doc_save(product2))
    print(esLib.index_mget("product_new",ids=[44,34]))
    print("update:",esLib.doc_update(product1,summary="hello world"))
    print(esLib.index_mget("product_new",ids=[44,34]))
    #print("delete:",esLib.doc_delete(product1))
    print(esLib.index_mget("product_new",ids=[44,34]))
    #print("index_delete",esLib.index_delete("product_new"))
    
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
    
    s=esLib.Search()
    