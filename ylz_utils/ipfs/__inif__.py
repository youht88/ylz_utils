import ipfshttpclient2
from ipfshttpclient2 import DEFAULT_ADDR,DEFAULT_BASE

class IpfsLib():    
    def __init__(self,addr:str=DEFAULT_ADDR,base:str=DEFAULT_BASE):
        print("正在初始化IPFS...")
        self.client = ipfshttpclient2.connect(addr,base)
        self.add_bytes = self.client.add_bytes
        self.add_json = self.client.add_json
        self.add_str = self.client.add_str
        self.add = self.client.add
        self.get = self.client.get
        self.get_json = self.client.get_json
        self.id = self.client.id
        self.cat = self.client.cat
        self.ls = self.client.ls
        self.pubsub = self.client.pubsub
        self.key = self.client.key



    