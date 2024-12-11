from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

# CUDA_VISIBLE_DEVICES="0" python main.py --data  --model SmolLM-1.7B --time_str 03_26_00_00_00

class SmolLM(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer.encode(conversation, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs)
        return self.tokenizer.decode(outputs[0])

model_core = SmolLM
