import base64
import httpx
from langchain_core.messages import HumanMessage

class HumanMessageWithImage(HumanMessage):
    def __init__(self,content:str=None,image:str|list[str]=None):
        img_b64 = []
        if image:
            images = image if type(image)==list else [image]
            for image in images:
                if image.startswith("https://") or image.startswith("http://"):
                    #img_b64.append(base64.b64encode(httpx.get(image).content).decode("utf8"))
                    img_b64.append(image)
                elif image.startswith("file://"):
                    #with open(image.rpartition("://")[2],"rb") as f:
                    #    img_b64.append(base64.b64encode(f.read()).decode("utf8"))
                    img_b64.append(image)
                else:
                    #原始的base64编码
                    img_b64.append(f"data:image;base64,{image}")
        content_with_image = [{"type":"text","text":content}]
        if img_b64:
            content_with_image.extend(
               [{"type":"image","image":image} for image in img_b64]
            )
        super().__init__(content_with_image)