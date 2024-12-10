import os
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from evals.utils import remove_image_token
from models.base import ModelWrapper
from configs import DATASET2DEFAULT_IMAGE_TOKEN

class QwenVL(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args).eval()

    def preprocess_vqa(self, data):
        instruction = remove_image_token(
            instruction=data['prompt_instruction'],
            tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']]
        )

        return [{'image': data['image_path'][0]}, {'text': instruction}]

    def generate_vqa(self, conversation, **kwargs):
        query = self.tokenizer.from_list_format(conversation)
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        return response

    def get_llm(self):
        return self.model.transformer

model_core = QwenVL
