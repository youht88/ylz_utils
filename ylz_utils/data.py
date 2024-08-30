from enum import Enum
import logging
import textwrap
from typing import List, Dict, Any, Literal, Tuple
import re
from urllib.parse import urlparse, urlunparse
import sys
import time
import itertools
import threading

class Spinner():
    def __init__(self,cursor=['|', '/', '-', '\\']):
        self.cursor = cursor
        self.stop_event = threading.Event()
    def spinning_cursor(self,duration, stop_event):
                spinner = itertools.cycle(self.cursor)
                end_time = time.time() + duration
                while time.time() < end_time:
                    if stop_event.is_set():
                        break
                    sys.stdout.write(next(spinner))
                    sys.stdout.flush()
                    sys.stdout.write('\b')  # 删除上一个字符
                    time.sleep(0.1)  # 控制动画的速度   
    def start(self):
        self.spinner_thread = threading.Thread(target=self.spinning_cursor, args=(float('inf'), self.stop_event))
        self.spinner_thread.start()
    
    def end(self):
        self.stop_event.set()
        self.spinner_thread.join()

class Color: 
    BLACK = '\033[30m'
    WHITE = '\033[37m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LRED = '\033[91m'
    LGREEN = '\033[92m'
    LYELLOW = '\033[93m'
    LBLUE = '\033[94m'
    LMAGENTA = '\033[95m'
    LCYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DARK = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    DELETE = '\033[9m'
ColorMap = {
    "black":Color.BLACK,"white":Color.WHITE,
    "red":Color.RED,"lred":Color.LRED,
    "green":Color.GREEN,"lgreen":Color.LGREEN,
    "yellow":Color.YELLOW,"lyellow":Color.LYELLOW,
    "blue":Color.BLUE,"lblue":Color.LBLUE,
    "magenta":Color.MAGENTA,"lmagenta":Color.LMAGENTA,
    "cyan":Color.CYAN,"lcyan":Color.LCYAN,
    "reset":Color.RESET,"bold":Color.BOLD,"dark":Color.DARK,"italic":Color.ITALIC,
    "underline":Color.UNDERLINE,"delete":Color.DELETE
}
class MathLib:
    pass

class StringLib:
    @classmethod
    def black(cls,text):
        return f"{Color.BLACK}{text}{Color.RESET}"
    @classmethod
    def white(cls,text):
        return f"{Color.WHITE}{text}{Color.RESET}"
    @classmethod
    def red(cls,text):
        return f"{Color.RED}{text}{Color.RESET}"
    @classmethod
    def green(cls,text):
        return f"{Color.GREEN}{text}{Color.RESET}"
    @classmethod
    def yellow(cls,text):
        return f"{Color.YELLOW}{text}{Color.RESET}"
    @classmethod
    def blue(cls,text):
        return f"{Color.BLUE}{text}{Color.RESET}"
    @classmethod
    def magenta(cls,text):
        return f"{Color.MAGENTA}{text}{Color.RESET}"
    @classmethod
    def cyan(cls,text):
        return f"{Color.CYAN}{text}{Color.RESET}"
    @classmethod
    def lred(cls,text):
        return f"{Color.LRED}{text}{Color.RESET}"
    @classmethod
    def lgreen(cls,text):
        return f"{Color.LGREEN}{text}{Color.RESET}"
    @classmethod
    def lyellow(cls,text):
        return f"{Color.LYELLOW}{text}{Color.RESET}"
    @classmethod
    def lblue(cls,text):
        return f"{Color.LBLUE}{text}{Color.RESET}"
    @classmethod
    def lmagenta(cls,text):
        return f"{Color.LMAGENTA}{text}{Color.RESET}"
    @classmethod
    def lcyan(cls,text):
        return f"{Color.LCYAN}{text}{Color.RESET}"
    @classmethod
    def bold(cls,text):
        return f"{Color.BOLD}{text}{Color.RESET}"
    @classmethod
    def dark(cls,text):
        return f"{Color.DARK}{text}{Color.RESET}"
    @classmethod
    def italic(cls,text):
        return f"{Color.ITALIC}{text}{Color.RESET}"
    @classmethod
    def underline(cls,text):
        return f"{Color.UNDERLINE}{text}{Color.RESET}"
    @classmethod
    def delete(cls,text):
        return f"{Color.DELETE}{text}{Color.RESET}"
    @classmethod
    def color(cls,text,style:\
              List[Literal["black","white","red","lred","green","lgreen","yellow","lyellow",\
                           "blue","lblue","magenta","lmagenta","cyan","lcyan", \
                           "bold","dark","italic","underline","delete"]] = ["yellow"]):
        if not style:
            return text
        for item in style:
            text = f"{ColorMap[item]}{text}{Color.RESET}"
        return text
    @classmethod
    def logging_in_box(cls,text, char="=", console_width:int=80, print_func = logging.info):
        """
        将传入的字符串用“=”字符串框起来在console打印出来，支持多行文本，= 对齐，
        并考虑了字符串在控制台中的实际显示宽度。

        Args:
            text: 要打印的字符串，可以包含多行。
            console_width: 控制台的宽度，默认为 80 个字符。
        """

        lines = text.splitlines()

        # 计算边框长度，考虑控制台宽度
        max_line_length = max(len(line) for line in lines)
        border_length = min(max_line_length + 4, console_width)

        # 打印上边框
        print_func(char * border_length)

        for line in lines:
            # 截断过长的行
            wrapped_lines = textwrap.wrap(line, width=console_width - 4)

            for wrapped_line in wrapped_lines:
                # 计算需要填充的空格数
                padding_space = border_length - len(wrapped_line) - 4
                padding = " " * padding_space

                # 打印带边框的文本行
                print_func(f"{char} {wrapped_line}{padding} {char}")

        # 打印下边框
        print_func(char * border_length)
    
class UrlLib:
    @classmethod
    def urlify(cls, address) -> tuple[str,str]:
        address = address.strip()
        # 解析URL
        parsed_url = urlparse(address)

        # 默认协议为 http
        if not parsed_url.scheme:
            address = 'http://' + address
            parsed_url = urlparse(address)

        # 判断地址类型
        url_type = None
        if parsed_url.scheme == 'file':
            url_type = 'file'
        elif parsed_url.scheme in ['http', 'https']:
            url_type = 'web'
        else:
            url_type = 'unknown'

        return address, url_type
    
    @classmethod
    def strip_protocol(cls, url):
        parsed_url = urlparse(url)
        # 构建去掉协议的URL
        stripped_url = parsed_url.netloc + parsed_url.path
        if parsed_url.query:
            stripped_url += '?' + parsed_url.query
        if parsed_url.fragment:
            stripped_url += '#' + parsed_url.fragment
        return stripped_url


class JsonLib:
    @classmethod
    def find_key_value_path(cls, json_data, target_key_path):
        # target_key_path 可以是以下类型
        # [] 表示数组占位
        # [2] 表示仅第3个数组占位
        # * 表示任何非数组的key占位
        # 其他字符串，表示key
        # example:  name.*.address , public.[].note.[1].local
        keys = target_key_path.split(".")
        keys.reverse()
        results = []
        def _search(data,current_path=[]):
            if isinstance(data,list):
                for index,item in enumerate(data):
                    current_path.append(index)
                    _search(item,current_path)
                    current_path.pop()
            elif isinstance(data,dict):
                for dict_key in data:
                    current_path.append(dict_key)
                    _search(data[dict_key],current_path)
                    current_path.pop()
            else:
                reversed_path = current_path.copy()
                if len(keys) > len(reversed_path):
                    return
                reversed_path.reverse()
                #print(data,keys,reversed_path)
                zip_keys = zip(keys,reversed_path)
                use = True
                for item in zip_keys:
                    key = item[0]
                    path = item[1]
                    if isinstance(path,int):
                        if key=="[]":
                            continue
                        elif re.match(r"\[(\d+)\]",key):
                            num = int(re.findall(r"\[(\d*)\]",key)[0])
                            if num != path:
                                use = False
                                break
                            else:
                                continue
                    if key=="*":
                        continue
                    if key != path:
                        use = False
                        break
                if use:
                    results.append({"value":data,"path":current_path.copy()})
        _search(json_data)
        return results                        
    @classmethod
    def find_key_value_path_old(cls, json_data, target_key_path,no_dict = True):
        """
        在 JSON 数据中查找特定路径下的 key 对应的值，并返回带有 JSON path 的结果。

        Args:
            json_data: 要搜索的 JSON 数据 (字典或列表)。
            target_key_path: 要查找的 key 路径，使用 "." 分隔层级，
                             例如 "address.city" 或 "phoneNumbers.[].city" 或 "bus.[].b.[].b1"。
            no_dict: 如果为True则只会查找最终的值，会忽略字典。如果为False找到匹配的字典不会再深入到字典里检索。默认为True
        Returns:
            一个列表，包含所有找到的 key 的值和路径信息。
            每个元素是一个字典，包含以下键值对：
                - "value": 找到的值
                - "path": 值在 JSON 数据中的路径，例如 ["phoneNumbers", 0, "city"]

        例如：
            [
                {"value": "Anytown", "path": ["address", "city"]},
                {"value": "Hometown", "path": ["phoneNumbers", 0, "city"]},
                {"value": "Workcity", "path": ["phoneNumbers", 1, "city"]}
            ]
        """

        keys = target_key_path.split(".")
        old_keys = "/".join(map(lambda key:'@[]@' if re.match(r'\[.*\]',key) else f"@{key}@",keys))
        results: List[Dict[str, Any]] = []

        def _process_keys(data, keys, current_path: List):
            if not keys:
                new_path = current_path.copy()
                new_keys = "/".join(map(lambda key:'@[]@' if re.match(r'^\d+$',str(key)) else f"@{str(key)}@",new_path))
                if (not isinstance(data, list)) and (not isinstance(data, dict)):
                    if new_keys.endswith(old_keys):
                        #results.append({"value":data,"path": new_path,"o1":old_keys,"n1":new_keys})
                        results.append({"value":data,"path": new_path})
                    return
                else:
                    if no_dict:
                        keys = target_key_path.split(".")
                    else:
                        if new_keys.endswith(old_keys):
                            #results.append({"value":data,"path": new_path,"o2":old_keys,"n2":new_keys})
                            results.append({"value":data,"path": new_path})
                        return
            key = keys[0]
            if key==[]:
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        current_path.append(i)
                        _process_keys(item, keys[1:], current_path)
                        current_path.pop()
            elif re.match(r"\[(\d+)\]",key):
                num = int(re.findall(r"\[(\d+)\]",key)[0])
                if isinstance(data, list):
                    if num < len(data):
                        current_path.append(num)
                        _process_keys(data[num], keys[1:], current_path)
                        current_path.pop()
            else:
                if isinstance(data, dict):
                    if key in data:
                        current_path.append(key)
                        _process_keys(data[key], keys[1:], current_path)
                        current_path.pop()
                    else:
                        for data_key in data.keys():
                            current_path.append(data_key)
                            _process_keys(data[data_key], keys, current_path)
                            current_path.pop()
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        current_path.append(i)
                        _process_keys(item, keys, current_path)
                        current_path.pop()
                else:
                    new_path = current_path.copy()
                    new_keys = "/".join(map(lambda key:'@[]@' if re.match(r'^\d+$',str(key)) else f'@{str(key)}@',new_path))
                    if new_keys.endswith(old_keys):
                        #results.append({"value":data,"path": new_path,"o3":old_keys,"n3":new_keys})
                        results.append({"value":data,"path": new_path})
        _process_keys(json_data, keys, [])
        return results
    @classmethod
    def update_json_by_path(cls, json_data:dict, updates=[], func=None)->dict:
        # updates = [
        #   {"value":"abc","path":["a","b",1,"c"]}
        #   {"value":"xyz","path":["x","0",y,"z"]}
        #   ...
        # ]
        # 将用func("abc") 更新a.b[1].c
        #    func("xyz") 更新x[0].y.z
        def _update_json_recursive(data, path, value):
            """递归地更新 JSON 数据。"""
            if not path:  # 路径为空，直接返回
                return

            key = path[0]

            if isinstance(data, list) and isinstance(key, int):
                if key >= len(data):
                    data.extend([None] * (key - len(data) + 1))
                if len(path) == 1:
                    data[key] = value
                else:
                    if data[key] is None:
                        data[key] = {}
                    _update_json_recursive(data[key], path[1:], value)
            elif isinstance(data, dict):
                if len(path) == 1:
                    data[key] = value
                else:
                    if key not in data:
                        data[key] = {}
                    _update_json_recursive(data[key], path[1:], value)
            else:
                raise ValueError(f"Unsupported type for key '{key}' at path: {path}")    
        for update in updates:
            value = update['value']
            if func:
                value = func(value)
            path = update['path']
            _update_json_recursive(json_data, path, value)
        return json_data


if __name__ == "__main__":
    json_data = {
        "name": "John",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "zip": "12345"
        },
        "p":{
          "a":{"p":5},"b":{"p":6},"c":{"p":7}
        },
        "phoneNumbers": [
            {"type": "home", "number": "555-555-1212", "city": "Hometown"},
            {"type": "work", "number": "555-555-1213", "city": "Workcity"}
        ],
        "bus": [
            {"p":888},
            {"a": [1,3,5], "b": {"b1": [1,{"x":1}],"p":999}},
            {"a": 2, "b": [{"bus": [3,{"y":2}]}]}
        ]
    }
    
    # 测试用例
    results = JsonLib.find_key_value_path(json_data, "city")
    print(f"\n\n结果1 =====> {results}")  # 应该输出：['Anytown', ['Hometown', 'Workcity']]

    
    results = JsonLib.find_key_value_path(json_data, "*.*.p")
    # results = JsonLib.find_key_value_path(json_data, "a",no_dict=False)
    print(f"\n\n结果2 ===>: {results}")  

    # results = JsonLib.update_json_by_path(json_data,results,lambda x:x*2)
    # print("*"*20,"\n",results)

    