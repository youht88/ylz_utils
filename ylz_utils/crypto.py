import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import uuid
import base64

class HashLib():
    @classmethod
    def md5(cls,obj:str|bytes):
        m = hashlib.md5()
        if isinstance(obj,bytes):
            m.update(obj)
        else:
            m.update(obj.encode('utf-8'))   
        return m.hexdigest()
    @classmethod
    def sha256(cls,obj:str|bytes):
        m = hashlib.sha256()
        if isinstance(obj,bytes):
            m.update(obj)
        else:
            m.update(obj.encode('utf-8'))
        return m.hexdigest()

class CryptoLib():
    @classmethod
    def getnode(cls):
        return str(uuid.getnode())
    @classmethod
    def generate_key_pair(cls):
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        private_key_hex = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        private_key_base64 = base64.b64encode(private_key_hex).decode('utf-8')
        public_key_hex = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        public_key_base64 = base64.b64encode(public_key_hex).decode('utf-8')
        return private_key_base64, public_key_base64

    # 签名消息
    @classmethod
    def sign_message(cls,private_key_base64:str, message:str)->str:
        message = message.encode()
        private_key_bytes = base64.b64decode(private_key_base64)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        signature = private_key.sign(message)
        signature_base64 = base64.b64encode(signature).decode('utf-8')
        return signature_base64

    # 验证签名
    @classmethod
    def verify_signature(cls,public_key_base64:str, message:str, signature_base64:str):
        message = message.encode()
        public_key_bytes = base64.b64decode(public_key_base64)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        try:
            signature = base64.b64decode(signature_base64)
            public_key.verify(signature, message)
            return True
        except Exception as e:
            return False

if __name__ == '__main__':
    # 生成密钥对,并将publi_key分发给用户
    private_key_base64,public_key_base64 = CryptoLib.generate_key_pair()
    print("private key=",private_key_base64)
    print("public key=",public_key_base64)
    # 用户端生成code，并发给SDK所有者
    code = CryptoLib.getnode()
    print("code=",code)
    # SDK所有者根据code生成sign，并发给用户
    sign = CryptoLib.sign_message(private_key_base64,code)
    print("sign=",sign)
    # 用户端验证
    assert(code==CryptoLib.getnode())
    assert(CryptoLib.verify_signature(public_key_base64,code,sign))
    print("验证通过")