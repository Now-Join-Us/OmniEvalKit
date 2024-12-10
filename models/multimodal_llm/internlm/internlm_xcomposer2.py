import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evals.utils import place_begin_image_token
from models.base import ModelWrapper
from configs import DATASET2DEFAULT_IMAGE_TOKEN


class XComposer2(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model_path = model_path

        if '4bit' in model_path:
            import auto_gptq
            from auto_gptq.modeling import BaseGPTQForCausalLM
            class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
                layers_block_name = "model.layers"
                outside_layer_modules = [
                    'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
                ]
                inside_layer_modules = [
                    ["attention.wqkv.linear"],
                    ["attention.wo.linear"],
                    ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                    ["feed_forward.w2.linear"],
                ]
            auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
            torch.set_grad_enabled(False)
            self.model = InternLMXComposer2QForCausalLM.from_quantized(model_path, **model_args).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
        else:
            if 'internlm-xcomposer2-7b' != model_path:
                torch.set_grad_enabled(False)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

        self.max_image = 1 if 'vl' in model_path else 2

    def preprocess_vqa(self, data):
        instruction = place_begin_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens=[f"<ImageHere>" for _ in range(len(data['image_path']))],
            leaved_token_num=self.max_image
        )

        count = 0

        images = data['image_preloaded'][:self.max_image]
        image = torch.stack(images)

        return instruction, image

    def generate_vqa(self, conversation, **kwargs):
        query, image = conversation
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=query, image=image, history=[], do_sample=False)

        return response

    def get_llm(self):
        pass

model_core = XComposer2
