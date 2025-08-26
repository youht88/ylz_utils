
import os
import base64
from cloudflare import Cloudflare
from ylz_utils.config import Config

class CloudflareLib:
    '''
    Cloudflare worker ai 库
    '''
    def __init__(self,api_key=None,account_id=None):
        self.api_token = api_key or os.environ.get("CLOUDFLARE_API_KEYS")
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.client = Cloudflare(api_token=self.api_token)

    def gen_image_base64(self,prompt,*,model_name=None):
        ''' 生成图像，默认调用@cf/black-forest-labs/flux-1-schnell模型，返回base64编码'''
        _response = self.client.ai.run(
            model_name = model_name or "@cf/black-forest-labs/flux-1-schnell",
            account_id=self.account_id,
            prompt=prompt,
        )
        return _response["image"]

    def gen_image_raw(self,prompt,*,model_name=None):
        ''' 生成图像，默认调用@cf/black-forest-labs/flux-1-schnell模型，返回raw原始数据'''
        _response = self.gen_image_base64(prompt,model_name=model_name)
        return base64.b64decode(_response)

    def gen_image_file(self,prompt,*,file_name,model_name=None):
        ''' 生成图像，默认调用@cf/black-forest-labs/flux-1-schnell模型，保存到文件'''
        _response = self.gen_image_raw(prompt,model_name=model_name)
        with open(file_name,"wb") as f:
            f.write(_response)


if __name__ == '__main__':
    Config.init('ylz_utils')
    cloudflare = CloudflareLib()
    PROMPT = """
    Cinematic medium close-up of a striking Chinese woman with sharp, angular facial features and jet-black hair in a sleek wet-look bob—strands clinging smoothly to her jawline, one loose tendril brushing her high cheekbone. Her skin is fresh, matte, and entirely free of oily sheen: no shine on the T-zone, forehead, nose, or cheeks, maintaining a clean, natural complexion with refined texture. She wears a structured midnight-blue satin cocktail dress with sharp shoulder pads and a thigh-high slit; the fabric catches cool silvery neon light, glinting softly to reveal faint neckline stitching. Backdrop: a moody, rain-soaked 1950s jazz club at midnight. Style: inspired by David Fincher’s atmospheric tones and classic film noir. 8K resolution, narrow depth of field (sharp face, blurred misty background), subtle film grain for gritty authenticity, and a desaturated palette (rich deep blues, ambers, neon accents) to amplify the retro-noir mood.    """
    cloudflare.gen_image_file(PROMPT,file_name="image.jpeg")
    print("file saved.")