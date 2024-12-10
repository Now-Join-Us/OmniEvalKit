from models.base import ModelWrapper
from transformers import pipeline

class Stablebeluga2(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

        prompt = f"{system_prompt}### User: {conversation}\n\n### Assistant:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

model_core = Stablebeluga2
