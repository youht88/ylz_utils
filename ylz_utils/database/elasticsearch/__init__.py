from datetime import datetime
from elasticsearch_dsl import Index as dsl_Index
from elasticsearch_dsl import connections, Field,Document, Date, Nested, Boolean, \
    analyzer, InnerDoc, Completion, Keyword, Text,Integer,Long,Double,Float,\
    DateRange,IntegerRange,FloatRange,IpRange,Ip,Range
from elasticsearch_dsl import Search,Q,FacetedSearch,MultiSearch
from elasticsearch_dsl.query import MultiMatch
from typing import Optional,Union,Literal
from elasticsearch.helpers import bulk
import pandas as pd
from ylz_utils.config import Config
import re

class ESLib():
    using = None
    analyzer='ik_max_word'
    search_analyzer='ik_smart'
    indexes = {}
    def __init__(self,hosts:str=None,
                 es_user:str=None ,
                 es_password:str=None ,
                 using:str = "es",
                 verify_certs=False,
                 ssl_show_warn=False,
                 analyzer='ik_max_word',search_analyzer='ik_smart'):
        self.config = Config()
        self.using = using
        self.analyzer = analyzer
        self.search_analyzer = search_analyzer
        connections.create_connection(
            hosts=hosts or self.config.get("DATABASE.ES.HOST"),
            basic_auth=(es_user or self.config.get("DATABASE.ES.USER"),es_password or self.config.get("DATABASE.ES.PASSWORD")),
            alias=using,
            verify_certs=verify_certs,
            ssl_show_warn=ssl_show_warn
        )
        self.client = connections.get_connection(alias=using)
        self.bulk = bulk
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
        wrapped_fields : dict[str,Field]= {}
        for key in raw_fields:
            field_type = raw_fields[key].get("type")
            multi = raw_fields[key].get("multi")
            required = raw_fields[key].get("requried")
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
                innerdoc_cls = type(innerdoc_name, (InnerDoc,), innerdoc_wrapped_fields)
                setattr(self,innerdoc_cls.__name__,innerdoc_cls) 
                #wrapped_fields[key] = innerdoc_cls() ????
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
    def get_search(self,index_name):
        return Search(using = self.using,index=index_name)
    def index_search(self,index_name,q:Q=None,s:Search|None=None,start=0,end=9):
        if not s:
            s = Search(using = self.using,index=index_name)
        if q:
            s=s.query(q)
        
        res = s[start:end].execute()
        cls = self.indexes[index_name]
        results = [cls(**data.to_dict()) for data in res]
        #size = res.hits.total.value
        return results
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
    def save(self,index_name:str,records:pd.DataFrame|list[dict],ids:list[str]=[]):
        actions = []
        if isinstance(records,pd.DataFrame):
            records = records.to_dict(orient='records')
        for record in records:        
            if ids:
                _id = '_'.join([str(record[key]) for key in ids])
                actions.append({"index":{"_index":index_name,"_id":_id}})
            else:
                actions.append({"index":{"_index":index_name}})
            actions.append(record)
        results = self.client.bulk(body=actions,refresh='wait_for')
        return results    
    def search(self,index_name:str,query):
        results  = self.client.search(index=index_name,body=query) 
        if results.get('hits'):
            data = [item['_source'] for item in results['hits']['hits']]
            return data
        else:
            return []
    def drop(self,index_name:str):
        result = None
        if self.client.indices.exists(index=index_name):
            result = self.client.indices.delete(index=index_name)
            print(f"已删除:{index_name}")
        else:
            print(f"{index_name}不存在!")
        return result
    def drop_multi(self,index_name_pattern:str):
        indices_to_delete = self.client.indices.get(index=index_name_pattern)
        result=[]
        for index in indices_to_delete:
            self.client.indices.delete(index=index, ignore=[400, 404])
            result.append(index)
        print(f"已删除:{str(result)}")
    def delete_by_query(self,index_name:str,query):
        results  = self.client.delete_by_query(index=index_name,body=query) 
        if results.get('deleted')>0:
            print(f"成功删除 {results['deleted']} 条文档")

    def update_by_query(self,index_name:str,query):
        results  = self.client.update_by_query(index=index_name,body=query) 
        if results.get('updated')>0:
            print(f"成功更新 {results['updated']} 条文档")
    def count(self,index_name:str,query):
        results  = self.client.count(index=index_name,body=query) 
        return results.get('count',0)
    def sql(self,query):
        result = self.client.sql.query(query=query) 
        return result  
    def sql_as_df(self,query):
        data = self.client.sql.query(query=query) 
        df = pd.DataFrame(data['rows'],columns=[item['name'] for item in data['columns']])
        return df  
if __name__ == '__main__':
    Config.init('ylz_utils')
    esLib = ESLib(using='es')
    class Product(Document):
        id = Integer()
        name = Text(analyzer='ik_max_word',search_analyzer='ik_smart',fields={'keyword':Keyword()})
        summary = Text(analyzer='ik_max_word',search_analyzer='ik_smart')
        solution_class = Text(analyzer='ik_max_word',search_analyzer='ik_smart')
        tags = Keyword()
        price = Float()
        published_from = Date()
        class Index():
            name = 'product'
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
    esLib.register_class(Product)
    # data = [ 
    #     {"id":1,"name":'碗',"summary":'镶边且印有小狐狸图案',"solution_class":'home',"tags":['little'],"price":20.2},
    #     {"id":2,"name":'筷子',"summary":'很多，黑色的高级筷子',"solution_class":'home',"tags":['black','multi'],"price":5.8},
    #     {"id":3,"name":'钥匙',"summary":'专用的多功能开锁神器',"solution_class":'tool',"tags":['useful'],"price":3.00},
    #     {"id":4,"name":'手机',"summary":'各种型号的大屏幕，大内存手机',"solution_class":'office',"tags":['good','multi'],"price":2888},
    #     {"id":5,"name":'打火机',"summary":'积压仓库，准备清仓',"solution_class":'tool',"tags":['cheap'],"price":0.1},
    #     {"id":6,"name":'鼠标',"summary":'白色无线鼠标，不需要更换电池，typeC充电',"solution_class":'office',"tags":['wireless','white'],"price":109},
    #     {"id":7,"name":'苹果',"summary":'20斤大苹果，包甜',"solution_class":'food',"tags":['red','good'],"price":2.2},
    #     {"id":8,"name":'电脑',"summary":'苹果imac M1，M2，M3系列',"solution_class":'office',"tags":['useful'],"price":4888},
    #     {"id":9,"name":'桌子',"summary":'减价销售，售完为止',"solution_class":'office',"tags":['cheap'],"price":260},
    #     {"id":10,"name":'足球',"summary":'官方认证，品质保障',"solution_class":'sport',"tags":['useful'],"price":60},
    #     {"id":11,"name":'面巾纸',"summary":'丝滑柔软',"solution_class":'home',"tags":['little','multi'],"price":40.5},
    #     {"id":12,"name":'水壶',"summary":'大容量，装水充足，自动报警，智能烧水',"solution_class":'home',"tags":['useful','cheap'],"price":99},
    #     {"id":13,"name":'自行车',"summary":'风华牌28寸，26寸都有',"solution_class":'home',"tags":['useful'],"price":120},
    #     {"id":14,"name":'手电筒',"summary":'4节电池，多功能',"solution_class":'tool',"tags":['useful'],"price":15},
    #     {"id":15,"name":'沙发',"summary":'墨绿色，豪华舒适',"solution_class":'home',"tags":['big','cheap'],"price":1500},
    #     {"id":16,"name":'香烟',"summary":'23条，吸烟有害健康',"solution_class":'food',"tags":['red'],"price":17.8},
    #     ]
    # products = []
    # for item in data:
    #     product = Product(
    #         meta={'id':item['id']},id=item['id'],name=item['name'],
    #         summary=item['summary'],solution_class=item['solution_class'],tags=item['tags'],price=item['price'])
    #     products.append(product)
    #     esLib.doc_save(product)
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
        "info":{"type":"innerdoc",
                "innerdoc":{"name":"Info",
                            "fields":{
                                "provider":{"type":"text"},
                                "price":{"type":"float"}
        }}}
    })
    # print("index_exists:",esLib.index_exists("product_new"))
    # docs = esLib.index_search("product_new")
    # product1 = ProductNew(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags=['Linabell'])
    # product1.info = esLib.Info(provider="abc",price=20.2)
    # print("*"*20,product1.to_dict())
    # product2 = ProductNew(meta={'id':34},id=1034,name='mouse',summary='a white mouse',solution_class='home',tags=['white','wireless'])
    # product2.info = esLib.Info(provider="xyz",price=30.4)
    # print("to_dict:",esLib.doc_dict(product1))
    # print("insert:",esLib.doc_save(product1))
    # print("insert:",esLib.doc_save(product2))
    # print(esLib.index_mget("product_new",ids=[44,34]))
    # print("update:",esLib.doc_update(product1,summary="hello world"))
    # print(esLib.index_mget("product_new",ids=[44,34]))
    # #print("delete:",esLib.doc_delete(product1))
    # print(esLib.index_mget("product_new",ids=[44,34]))
    # #print("index_delete",esLib.index_delete("product_new"))
    
    # # for doc in docs:
    # #     esLib.delete(doc)
    # #product.save(using="es")
    # #esLib.save("product_new")  
    # # product = Product(meta={'id':44},id=1044,name='soup',summary='bowl with a little fox',solution_class='home',tags='Linabell')
    # # product.publish_from = datetime.now()
    # # product.save(using="es")
    
    # # q = esLib.termQ("tags","Linabell")
    # # res,size = esLib.search("product",q) 
    # # if res:
    # #     print(res[0].name,size)
    # # else:
    # #     print("no result")
    
    # # s=esLib.get_search("product")
    # # s.query("term",solution_class="abc")
    # # res = s.execute()
    # # print([(hit.meta.score , hit.name) for hit in res.hits])
    
    # results = esLib.index_search("product")
    # print([result.to_dict() for result in results])

    s = esLib.get_search("product")
    s.query("term",name="手机")
    results = esLib.index_search("product",s=s)
    print([result.to_dict() for result in results])