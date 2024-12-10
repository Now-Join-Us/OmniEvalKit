import os
import torch
from transformers import AutoModel, AutoTokenizer
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import remove_image_token


class MiniCPM_Llama3_V(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, **model_args).to(dtype=model_args['torch_dtype'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

    def preprocess_vqa(self, data):
        instruction = remove_image_token(
            instruction=data['prompt_instruction'],
            tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']]
        )

        msgs = [{'role': 'user', 'content': instruction}]

        conversation = (msgs, data['image_preloaded'][0])
        return conversation

    def generate_vqa(self, conversation):
        msgs, image = conversation
        response = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )
        return response

    def get_llm(self):
        return self.llm

model_core = MiniCPM_Llama3_V