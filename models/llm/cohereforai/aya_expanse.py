from models.base import ModelWrapper
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM


# CUDA_VISIBLE_DEVICES="0" python main.py --data  --model aya-expanse-8b --time_str 03_26_00_00_00

class aya_expanse(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        # Format the message with the chat template
        messages = [{"role": "user", "content": conversation}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.3,
            )

        return tokenizer.decode(gen_tokens[0])

model_core = aya_expanse
