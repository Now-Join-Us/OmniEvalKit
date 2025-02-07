from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

class Yi_Chat(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {"role": "user", "content": conversation}
        ]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.model.device), eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=512)

        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

model_core = Yi_Chat
