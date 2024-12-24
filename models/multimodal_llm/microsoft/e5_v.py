from models.base import ModelWrapper
from configs import DATA_PATH

import os
import io

import torch
import torch.nn.functional as F
import requests
from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import oss2 as oss
from evals.utils import replace_image_token
from configs import DATASET2DEFAULT_IMAGE_TOKEN


class E5_V(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, **{k: v for k, v in model_args.items() if k != 'documents_file'}).to(torch.device('cuda'))

        if 'documents_file' in model_args.keys():
            with open(os.path.join(DATA_PATH, model_args['documents_file']), 'r', encoding='utf-8') as f:
                mc_list = [i.strip() for i in f.readlines()]
            mc_prompts = [self.llama3_template.format(i) for i in mc_list]
            text_inputs = self.processor(mc_prompts, return_tensors="pt", padding=True).to(self.model.device)

            with torch.no_grad():
                self.text_embs = self.model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                self.text_embs = F.normalize(self.text_embs, dim=-1)

    def preprocess_vqa(self, data):
        instruction = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens='<image>',
            leaved_token_num=1
        )
        if '<image>' not in instruction:
            instruction =  '<image>\n' + instruction
        return instruction

    def generate_vqa(self, conversations, device, **kwargs):
        img_prompts = [self.llama3_template.format(self.preprocess_vqa(i)) for i in conversations if 'image_path' in i.keys()]
        images = [data['image_preloaded'][0] for data in conversations if 'image_preloaded' in data.keys()]

        scores = []
        if len(images) != 0:
            img_inputs = self.processor(img_prompts, images, return_tensors="pt", padding=True).to(self.model.device)

            with torch.no_grad():
                img_embs = self.model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                img_embs = F.normalize(img_embs, dim=-1)

            scores = img_embs @ self.text_embs.t()
            scores = scores.tolist()

        score_index = 0
        results = {}
        for i in conversations:
            if 'image_path' in i.keys():
                results[i['id']] = scores[score_index]
                score_index += 1
            else:
                results[i['id']] = None
        return results


model_core = E5_V
