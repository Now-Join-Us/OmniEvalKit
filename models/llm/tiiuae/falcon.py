from models.base import ModelWrapper
from transformers import pipeline

class Falcon(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_text_only(self, conversation, **kwargs):
        sequences = self.pipeline(
            conversation,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return sequences[0]['generated_text']

model_core = Falcon
