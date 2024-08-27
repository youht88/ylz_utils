from __future__ import annotations
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis

class FluxLib():
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def init(self):
        api_keys = self.config.get('FLUX.DASHSCOPE.API_KEYS')        
        self.model = self.config.get('FLUX.DASHSCOPE.MODEL') or 'flux-schnell'
        if api_keys:
            api_key = self.langchainLib.split_keys(api_keys)[0]
            self.api_key=api_key
        else:
            raise Exception("请先设置FLUX.DASHSCOPE.API_KEYS")
    def gen(self,input_prompt) -> str:
        self.init()
        rsp = ImageSynthesis.call(model=self.model,
                                api_key= self.api_key,
                                prompt=input_prompt,
                                size='1024*1024')
        if rsp.status_code == HTTPStatus.OK:
            print(rsp.output)
            print(rsp.usage)
            # save file to current directory
            for result in rsp.output.results:
                file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                with open('./%s' % file_name, 'wb+') as f:
                    f.write(requests.get(result.url).content)
                return file_name
        else:
            print('Failed, status_code: %s, code: %s, message: %s' %
                (rsp.status_code, rsp.code, rsp.message))
            return ""

    def agen(self,input_prompt):
        self.init()
        rsp = ImageSynthesis.async_call(model=self.model,
                                        api_key=self.api_key,
                                        prompt=input_prompt,
                                        size='1024*1024')
        if rsp.status_code == HTTPStatus.OK:
            print(rsp.output)
            print(rsp.usage)
        else:
            print('Failed, status_code: %s, code: %s, message: %s' %
                (rsp.status_code, rsp.code, rsp.message))
        status = ImageSynthesis.fetch(rsp)
        if status.status_code == HTTPStatus.OK:
            print(status.output.task_status)
        else:
            print('Failed, status_code: %s, code: %s, message: %s' %
                (status.status_code, status.code, status.message))

        rsp = ImageSynthesis.wait(rsp)
        if rsp.status_code == HTTPStatus.OK:
            print(rsp.output)
        else:
            print('Failed, status_code: %s, code: %s, message: %s' %
                (rsp.status_code, rsp.code, rsp.message))


if __name__ == '__main__':
    prompt = "Eagle flying freely in the blue sky and white clouds"
    prompt_cn = "一只飞翔在蓝天白云的鹰"
    fluxLib = FluxLib(None)
    fluxLib.gen(prompt)
    fluxLib.agen(prompt_cn)