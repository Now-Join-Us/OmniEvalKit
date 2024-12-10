from models.base import ModelWrapper
from transformers import pipeline

class Neuraldaredevil(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate_text_only(self, conversation, **kwargs):
        messages = [{"role": "user", "content": conversation}]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

model_core = Neuraldaredevil
