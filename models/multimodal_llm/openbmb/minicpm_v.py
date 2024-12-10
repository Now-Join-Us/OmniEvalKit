import os
import torch
from transformers import AutoModel, AutoTokenizer
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import remove_image_token


class MiniCPM_V(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, **model_args).to(dtype=model_args['torch_dtype'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
        self.num_beams = 1 if 'MiniCPM-V' in model_path and 'MiniCPM-V-' not in model_path else 3

    def preprocess_vqa(self, data):
        instruction = remove_image_token(
            instruction=data['prompt_instruction'],
            tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']]
        )

        msgs = [{'role': 'user', 'content': instruction}]

        conversation = (msgs, data['image_preloaded'][0])
        return conversation

    def generate_vqa(self, conversation):
        image = conversation[-1]

        conversation = conversation[0]

        default_kwargs = dict(
            max_new_tokens=1024,
            sampling=False,
            num_beams=self.num_beams
        )
        res, _, _ = self.model.chat(
            image=image,
            msgs=conversation,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        return res

    def get_llm(self):
        return self.llm

model_core = MiniCPM_V