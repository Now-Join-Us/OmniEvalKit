import os
import openai
from models.base import ModelWrapper

url = 'https://openxlab.org.cn/gw/alles-apin-hub/v1/claude/v1/text/chat'
headers = {
    'alles-apin-token': '',
    'Content-Type': 'application/json'
}

class Claude(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        self.headers = headers
        self.temperature = model_args['temperature']
        self.max_tokens = model_args['max_tokens']

        if 'api_key' not in model_args:
            self.key = model_args['api_key']
        else:
            self.key = os.environ.get('ALLES', '')

        self.model = model_path

    def generate_text_only(self, conversation, **kwargs):
        data = [{"role": "user", "content": conversation}]

        success = False
        cnt = 0
        response = ''
        while not success and cnt < 5:
            try:
                response = openai.ChatCompletion.create(model=self.model, messages=data)
                response = response["choices"][0]["message"]["content"]
                response = response.replace('\n', '')

            except Exception as e:
                # print(e)
                cnt += 1
                response = ""
                continue

        return response

model_core = Claude