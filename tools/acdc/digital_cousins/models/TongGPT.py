import os
from openai import AzureOpenAI, OpenAI


# from utils import local_image_to_data_url, resize, extract_json



class TongGPT():
    def __init__(self, MODEL="gpt-35-turbo-0125",REGION="eastus2",api_key=None):
        super().__init__()
        self.REGION =  REGION
        self.MODEL = MODEL
        self.API_KEY = api_key

        # Check for API type from environment or default to openai
        # Use standard OpenAI API by default for better compatibility
        self.api_type = os.environ.get("OPENAI_API_TYPE", "openai").lower()
        self.api_version = "2024-02-01"
        self.init_client()

    def init_client(self):
        if self.api_type == "azure":
            API_BASE = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://api.tonggpt.mybigai.ac.cn/proxy")
            self.ENDPOINT = f"{API_BASE}/{self.REGION}"
            self.client = AzureOpenAI(
                api_key=self.API_KEY,
                api_version=self.api_version,
                azure_endpoint=self.ENDPOINT,
            )
        else:
            # Use standard OpenAI API
            self.client = OpenAI(api_key=self.API_KEY)
        return self.client
    
    def send_request(self,kw):
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": kw}
            ],
        )

        print(response.model_dump_json(indent=2))
        print(".....")
        print(response.choices[0].message.content)
        return response


class GPT4o(TongGPT):
    def __init__(self, MODEL="gpt-4-turbo-2024-04-09",REGION="eastus2",api_key=None):
        super().__init__(MODEL,REGION,api_key)

    def send_request(self,payload):

        response = self.client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload["temperature"],
            max_tokens=payload["max_tokens"],
        )
        return response
    

class GPT4V(GPT4o):
    def __init__(self, MODEL="gpt-4-vision-preview",REGION="australiaeast"):
        super().__init__(MODEL,REGION)


# if __name__=="__main__":
# #     tonggpt = TongGPT()
# #     response = tonggpt.send_request("Say Hello!")
# #     a = 1
# #     # gen_results(job_id)

#     from openai import AzureOpenAI


#     REGION = "southcentralus"
#     MODEL = "gpt-4-0125-preview"
#     API_KEY =  "e989ba33fe61d1251d2921132320c92c"

#     API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
#     ENDPOINT = f"{API_BASE}/{REGION}"


#     client = AzureOpenAI(
#     api_key=API_KEY,
#     api_version="2024-02-01",
#     azure_endpoint=ENDPOINT,
#     )

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "user", "content": "Say Hello."}
#         ],
#     )

#     print(response.model_dump_json(indent=2))
#     print(response.choices[0].message.content)