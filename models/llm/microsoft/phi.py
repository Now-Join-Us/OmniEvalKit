from models.base import ModelWrapper
from models.base import ModelWrapper

class Phi(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.tokenizer.pad_token = self.tokenizer.eos_token

model_core = Phi
