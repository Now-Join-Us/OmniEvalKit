from models.base import ModelWrapper

class QwenBase(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer(conversation, return_tensors='pt').to(self.model.device)
        pred = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response

model_core = QwenBase
