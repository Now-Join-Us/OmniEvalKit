import os
import torch
from transformers import AutoModel, AutoTokenizer
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import remove_image_token


class MiniCPM_V_2_6(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, **model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

    def preprocess_vqa(self, data):
        instruction = remove_image_token(
            instruction=data['prompt_instruction'],
            tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']]
        )

        msgs = [{'role': 'user', 'content': instruction}]

        return msgs, data['image_preloaded'][0]

    def generate_vqa(self, conversation):
        msgs, image = conversation

        response = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
        )
        return response

    def generate_ppl(self, conversation):
        pass
    def get_llm(self):
        return self.model

model_core = MiniCPM_V_2_6