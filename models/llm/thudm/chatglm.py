from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
from models.base import ModelWrapper

class ChatGLM(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        self.model_path = model_path

        if 'chatglm-6b' in model_path:
            self.model = AutoModel.from_pretrained(model_path, **tokenizer_args).half()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **model_args)

        else:
            super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        try:
            response = self.generate_with_chat(self.tokenizer, conversation)
        except ValueError as e:
            if 'increasing `max_length`' in str(e):
                response = ''

        return response

model_core = ChatGLM
