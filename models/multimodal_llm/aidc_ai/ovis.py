from models.base import ModelWrapper
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from evals.utils import place_begin_image_token
import os
from transformers import AutoModelForCausalLM
import torch

class Ovis(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        # --model_args torch_dtype:torch.bfloat16,multimodal_max_length:8192,trust_remote_code:True
        if 'Ovis-Clip-Llama3-8B' in model_path:
            model_path = 'AIDC-AI/Ovis-Clip-Llama3-8B'
        elif 'Ovis-Clip-Qwen1_5-7B' in model_path:
            model_path = 'AIDC-AI/Ovis-Clip-Qwen1_5-7B'
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()

    def preprocess_vqa(self, data):
        query = place_begin_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens=["<image>"],
            leaved_token_num=1,
            sep=' '
        )

        conversation = {
            'query': query,
            'image_preloaded': data['image_preloaded'][0]
        }

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        query, image = conversation['query'], conversation['image_preloaded']
        prompt, input_ids = self.conversation_formatter.format_query(query)
        input_ids = torch.unsqueeze(input_ids, dim=0).to(device=self.model.device)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).to(device=self.model.device)
        pixel_values = [
            self.visual_tokenizer.preprocess_image(image).to(
                dtype=self.visual_tokenizer.dtype,
                device=self.visual_tokenizer.device
            )
        ]

        with torch.inference_mode():
            kwargs = dict(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                do_sample=False,
                top_p=None,
                temperature=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=512,
                use_cache=True,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id
            )
            output_ids = self.model.generate(input_ids, **kwargs)[0]
            input_token_len = input_ids.shape[1]
            response = self.text_tokenizer.decode(output_ids[input_token_len:], skip_special_tokens=True)
            return response

    def get_llm(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.model.language_model

model_core = Ovis
