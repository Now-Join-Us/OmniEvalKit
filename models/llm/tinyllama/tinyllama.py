from models.base import ModelWrapper
from transformers import pipeline

class Tinyllama(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer)

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": conversation},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response = outputs[0]["generated_text"]

        return response

model_core = Tinyllama
