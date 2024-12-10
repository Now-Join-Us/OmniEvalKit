import torch
import os
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import replace_image_token

class CogVLM(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args).eval()
        self.kwargs = kwargs

        tokenizer_name = tokenizer_args.pop('tokenizer_name', False)
        if tokenizer_name:
            tokenizer_path = os.path.join(os.path.split(model_path)[0], tokenizer_name)
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
            gen_kwargs = {'max_length': 2048, 'do_sample': False}
            self.end_text_token = '</s>'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
            gen_kwargs = {'max_new_tokens': 2048, 'pad_token_id': 128002}
            self.end_text_token = '<|end_of_text|>'
        self.kwargs.update(gen_kwargs)

    def preprocess_vqa(self, data):
        instruction = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens='',
            leaved_token_num=1
        )

        conversation = (instruction, data['prompt_instruction'][0])

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        instruction, image = conversation
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=instruction, history=[], images=[image])  # chat mode

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[inputs['images'][0].to(self.model.device).to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        response = response.split(self.end_text_token)[0].strip()
        return response

    def get_llm(self):
        return None

model_core = CogVLM