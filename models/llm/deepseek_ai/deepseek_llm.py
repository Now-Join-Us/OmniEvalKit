from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from models.base import ModelWrapper

class DeepSeekChat(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {"role": "user", "content": conversation}
        ]
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=100)

        response = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return response

model_core = DeepSeekChat
