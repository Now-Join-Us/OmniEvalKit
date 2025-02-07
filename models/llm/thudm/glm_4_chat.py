from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper
import torch

class GLM4Chat(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.gen_kwargs = {"do_sample": True, "top_k": 1}

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": conversation}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs, max_new_tokens=512)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

model_core = GLM4Chat
