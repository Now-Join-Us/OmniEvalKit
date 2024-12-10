from transformers import GenerationConfig
from models.base import ModelWrapper

class OpenOcra(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        sys_prompt = "A chat."
        prompt = conversation

        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + sys_prompt + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format

        generation_config = GenerationConfig(
            max_length=self.max_length, temperature=1.1, top_p=0.95, repetition_penalty=1.0,
            do_sample=True, use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id,
            transformers_version="4.34.0.dev0")

        inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(self.model.device)
        outputs = self.model.generate(**inputs, generation_config=generation_config)

        response = self.tokenizer.batch_decode(outputs)[0]
        return response

model_core = OpenOcra
