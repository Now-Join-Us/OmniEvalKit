import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

from evals.utils import place_begin_image_token
from configs import DATASET2DEFAULT_IMAGE_TOKEN


class Monkey(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

        if 'Chat' in model_path:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token_id = self.tokenizer.eod_id

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args).eval()
        self.kwargs = kwargs

    def preprocess_vqa(self, data):
        instruction = place_begin_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens=[f"<img>{os.path.join(data['image_url'], i)}</img>" for i in data['image_path']],
            leaved_token_num=1
        )
        conversation = {
            'instruction': instruction,
        }
        return conversation

    def generate_vqa(self, conversation, **kwargs):
        input_ids = self.tokenizer(conversation['instruction'], return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        output_ids = self.model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
        )
        response = self.tokenizer.decode(
            output_ids[0][input_ids.size(1):].cpu(),
            skip_special_tokens=True
        ).strip()
        return response

model_core = Monkey