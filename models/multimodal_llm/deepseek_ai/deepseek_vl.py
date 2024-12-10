import os
import sys
import warnings

import torch
from transformers import AutoModelForCausalLM

from models.base import ModelWrapper
from evals.utils import replace_image_token
from configs import DATASET2DEFAULT_IMAGE_TOKEN

try:
    from deepseek_vl.models import VLChatProcessor
    from deepseek_vl.utils.io import load_pil_images
except ImportError:
    warnings.warn(
        'Please install deepseek_vl from https://github.com/deepseek-ai/DeepSeek-VL')
    sys.exit(-1)


class DeepSeekVL(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(torch.bfloat16)

        self.kwargs = {
            'max_new_tokens': 512,
            'do_sample': False,
            'use_cache': True
        }
        self.kwargs.update(kwargs)

    def preprocess_vqa(self, data):
        data['prompt_instruction'] = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens='<image_placeholder>',
            leaved_token_num=1
        )
        conversation = [
            dict(
                role='User',
                content=data['prompt_instruction'],
                images=[os.path.join(data['image_url'], data['image_path'][0])]
            ),
            dict(
                role='Assistant',
                content=''
            )
        ]

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
        prepare_inputs = prepare_inputs.to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs)
        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    def get_llm(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.model.language_model

model_core = DeepSeekVL
