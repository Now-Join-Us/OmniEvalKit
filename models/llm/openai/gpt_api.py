import openai
from models.base import ModelWrapper

class GPT(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        if 'api_key' not in model_args:
            print('api_key not found, please put api_key in model_args')
            raise Exception
        openai.api_key = model_args['api_key']

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

            except Exception as e:
                # print(e)
                cnt += 1
                response = ""
                continue

        return response
