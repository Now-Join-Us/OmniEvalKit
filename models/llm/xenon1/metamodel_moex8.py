from models.base import ModelWrapper
from transformers import AutoTokenizer, pipeline

class Moex(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs=model_args,
        )
        super().__init__(model=self.pipe.model, tokenizer=self.tokenizer)

    def generate_text_only(self, conversation, **kwargs):
        messages = [{"role": "user", "content": conversation}]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)[0]["generated_text"]
        return response

model_core = Moex
