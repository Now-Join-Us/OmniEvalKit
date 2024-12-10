import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

class QwenVLChat(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)

    def preprocess_vqa(self, data):
        conversation = self.tokenizer.from_list_format([
            {'image': os.path.join(data['image_url'], data['image_path'][0])},
            {'text': data['prompt_instruction']},
        ])

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        response, _ = self.model.chat(self.tokenizer, query=conversation, history=None)
        return response

    def get_llm(self):
        return self.transformer

model_core = QwenVLChat
