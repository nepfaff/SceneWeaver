# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI



import base64

class DeepSeek():
    """
    Simple interface for interacting with deepseek model
    """
    def __init__(self):
        super().__init__()
       
        API_BASE = "https://api.deepseek.com"
        api_key="sk-f6dc786ec7a34381b457b0cb3e1da4df"
        self.MODEL = "deepseek-reasoner"
        self.init_client(API_BASE,api_key)

    def init_client(self,API_BASE,api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url=API_BASE
        )
        return self.client
    
    def __call__(self, payload, verbose=False):
       
       
        response = self.send_request(payload)
        try:
            content = response.choices[0].message.content
        except:
            print(
                f"Got error while querying GPT-{self.version} API! Response:\n\n{response}"
            )
            return None
    
        return content
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def send_request(self, payload):
        response = self.client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            stream=False
        )
        
        print(response.model_dump_json(indent=2))
        print(".....")
        print(response.choices[0].message.content)
        return response
    
    def get_payload(self, prompting_text_system, prompting_text_user):
        text_dict_system = {"type": "text", "text": prompting_text_system}
        content_system = [text_dict_system]

        content_user = [{"type": "text", "text": prompting_text_user}]

        object_caption_payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user},
            ],
            # "temperature": 0,
            # "max_tokens": 4096,
        }
        return object_caption_payload

    def get_payload_scene_image(self, prompting_text_system, prompting_text_user,render_path=None):
        text_dict_system = {"type": "text", "text": prompting_text_system}
        content_system = [text_dict_system]

        if render_path is not None:
            imgs_base64 = self.encode_image(render_path) 
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{imgs_base64}"
                }
            }

            prompting_user_lst = prompting_text_user.split("SCENE_IMAGE")
            content_user = [{"type": "text", "text": prompting_user_lst[0]},
                            img_dict,
                            {"type": "text", "text": prompting_user_lst[1]}]
        else:
            content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            }
        ]

    
        object_caption_payload = {
            # "model": "gpt-4-turbo-2024-04-09",
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        return object_caption_payload
    
   