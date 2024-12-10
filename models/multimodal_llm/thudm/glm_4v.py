import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import replace_image_token

class GLM_4V(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_args
        ).eval()
        gen_kwargs = {'max_length': 2048, 'do_sample': False}
        gen_kwargs.update(kwargs)
        self.kwargs = gen_kwargs
        self.end_text_token = '<|endoftext|>'

    def preprocess_vqa(self, data):
        instruction = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens='',
            leaved_token_num=1
        )

        return (instruction, data['prompt_instruction'])

    def generate_vqa(self, conversation, **kwargs):
        instruction, image = conversation
        inputs = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'image': image, 'content': instruction}],
            add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        return response.split(self.end_text_token)[0]

    def get_llm(self):
        return None

model_core = GLM_4V