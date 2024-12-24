from models.base import ModelWrapper
from transformers import pipeline

class LLaMA3_1Base(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={k: v for k, v in model_args.items() if k != 'device_map'},
            device_map="auto"
        )

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": conversation},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][-1]

        return response

model_core = LLaMA3_1Base
