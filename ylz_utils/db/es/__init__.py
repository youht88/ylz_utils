from datetime import datetime
from elasticsearch_dsl import connections,Document,Date,Integer,Keyword,Text

class ESLib():
    def __init__(self):
        connections.create_connection(
            hosts=['https://localhost:9200'],
            basic_auth=('elastic','9HIMozq48xIP+PHTpRVP'),
            alias='es',
            verify_certs=False,
            ssl_show_warn=False
        )
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
Product.init(using='es')
product = Product(meta={'id':42},id=1042,name='bowl',summary='bowl with a little fox',solution_class='home',tags='Linabell')
product.publish_from = datetime.now()
product.save(using='es')

product = Product.get(id=42,using='es')