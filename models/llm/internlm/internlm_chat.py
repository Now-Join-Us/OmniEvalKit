from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

class InternLMChat(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        response = self.generate_with_chat(self.tokenizer, conversation)
        return response

model_core = InternLMChat
