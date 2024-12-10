from models.base import ModelWrapper

class QwenChat(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        response, history = self.model.chat(self.tokenizer, conversation, history=None)

        return response

model_core = QwenChat
