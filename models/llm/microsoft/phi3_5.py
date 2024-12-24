from models.base import ModelWrapper
from transformers import pipeline

class Phi3_5(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": conversation},
        ]

        generation_args = {
            "max_new_tokens": 512,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = self.pipe(messages, **generation_args)
        response = output[0]['generated_text']

        return response

model_core = Phi3_5
