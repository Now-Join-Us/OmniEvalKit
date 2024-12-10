import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import replace_image_token

class BunnyLlamma(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

    def preprocess_vqa(self, data):
        instruction = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens='<image>',
            leaved_token_num=1
        )

        return (instruction, data['image_preloaded'][0])

    def generate_vqa(self, conversation, **kwargs):
        instruction, image = conversation

        text_chunks = [self.tokenizer(chunk).input_ids for chunk in instruction.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(self.model.device)
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype, device=self.model.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0  # increase this to avoid chattering
        )[0]

        response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return response

    def get_llm(self):
        return self.model

model_core = BunnyLlamma
