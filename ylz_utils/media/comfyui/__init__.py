#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import base64


class ComfyuiLib:
    server_address = "127.0.0.1:8188"
    client_id = str(uuid.uuid4())
    def __init__(self,server_address=None,client_id=None):
        if server_address:
            self.server_address = server_address
        if client_id:
            self.client_id = client_id

    def queue_prompt(self,prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    
    # 读取本地图片并转为Base64
    def image_to_base64(self,image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    def get_image(self,filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(self,prompt_id):
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())
    def get_queue(self):
        with urllib.request.urlopen(f"http://{self.server_address}/queue") as response:
            return json.loads(response.read())
    def get_images(self,ws, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
                # bytesIO = BytesIO(out[8:])
                # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
                continue #previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

        return output_images

if __name__ == "__main__":
    with open("nunchaku-flux.1-kontext-dev.json") as f:
        prompt_text = f.read()

    prompt = json.loads(prompt_text)
    #set the text prompt for our positive CLIPTextEncode
    prompt["6"]["inputs"]["text"] = "Convert it to a photo of the girl from the back, with a beam of sunlight shining on her hair."

    comfyuiLib = ComfyuiLib("192.168.24.142:7860")
    ws = websocket.WebSocket()
    ws.connect(f"ws://{comfyuiLib.server_address}/ws?clientId={comfyuiLib.client_id}")
    print(comfyuiLib.get_queue())
    images = comfyuiLib.get_images(ws, prompt)
    ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
    #Commented out code to display the output images:

    for node_id in images:
        for image_data in images[node_id]:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            image.show()

