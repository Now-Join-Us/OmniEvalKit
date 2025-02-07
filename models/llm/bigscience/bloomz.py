from models.base import ModelWrapper
from transformers import pipeline

# bloomz-560m model_args: torch_dtype="auto", device_map="auto"
# bloomz-1b7 model_args: torch_dtype="auto", device_map="auto"

class Bloomz(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer.encode(conversation, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0])

model_core = Bloomz
