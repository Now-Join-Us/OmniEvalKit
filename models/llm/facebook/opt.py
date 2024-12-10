from transformers import pipeline
from models.base import ModelWrapper

class OPT(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate_text_only(self, conversation, **kwargs):
        response = self.pipe(conversation, max_length=1024)

        return response[0]['generated_text']

model_core = OPT
