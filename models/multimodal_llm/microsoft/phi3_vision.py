import os
import io
import json
import torch
import logging
import oss2 as oss
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from evals.utils import replace_image_token
from models.base import ModelWrapper
from configs import DATASET2DEFAULT_IMAGE_TOKEN

class Phi3Vision(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_args).eval()
        processor = AutoProcessor.from_pretrained(model_path, **tokenizer_args)
        self.model = model
        self.processor = processor
        self.kwargs = kwargs
        self.tokenizer = self.processor.tokenizer

    def preprocess_prompt_instruction(self, prompt_instruction, dataset_name, img_num):
        prompt_instruction = replace_image_token(
            instruction=prompt_instruction,
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN.get(dataset_name, ['<image>']),
            target_tokens=[f"<|image_{i + 1}|>" for i in range(img_num)],
            leaved_token_num=img_num
        )
        return [{'role': 'user', 'content': prompt_instruction}]

    def preprocess_vqa(self, data):
        img_num = len(data['image_path'])
        instruction = self.preprocess_prompt_instruction(data['prompt_instruction'], data['name'], img_num)

        return (instruction, data['image_preloaded'][0])

    def get_generated(self, instruction, image, **kwargs):
        prompt = self.processor.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(prompt, [image] if image is not None else None, return_tensors='pt').to(self.model.device)

        generation_args = {
            'max_new_tokens': 1000,
            'temperature': 0.0,
            'do_sample': False,
        }
        generation_args.update(self.kwargs)
        return inputs, generation_args

    def generate_vqa(self, conversation, **kwargs):
        instruction, image = conversation
        inputs, generation_args = self.get_generated(instruction, image, **kwargs)

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def get_llm(self):
        return self.model

model_core = Phi3Vision
