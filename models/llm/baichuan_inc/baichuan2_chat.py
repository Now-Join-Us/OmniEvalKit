from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from models.base import ModelWrapper

class Baichuan2Chat(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)

    def generate_text_only(self, conversation, **kwargs):
        messages = [{"role": "user", "content": conversation}]
        response = self.model.chat(self.tokenizer, messages)
        return response

model_core = Baichuan2Chat
