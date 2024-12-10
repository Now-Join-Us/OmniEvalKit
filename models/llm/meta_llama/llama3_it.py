from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

class LLaMA3Instruct(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": conversation},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        return response

model_core = LLaMA3Instruct
