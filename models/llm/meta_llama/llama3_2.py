from models.base import ModelWrapper
from transformers import pipeline

class LLaMA3_2(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            **{k: v for k, v in model_args.items() if k != 'device_map'}
        )
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer

    def generate_text_only(self, conversation, **kwargs):
        outputs = self.pipeline(
            conversation,
            max_new_tokens=512,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][-1]

        return response

model_core = LLaMA3_2
