from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

class Baichuan(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer(f'{conversation}->', return_tensors='pt').to(self.model.device)

        pred = self.model.generate_text_only(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response

model_core = Baichuan
