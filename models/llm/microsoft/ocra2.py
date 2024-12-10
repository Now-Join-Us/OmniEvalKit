from models.base import ModelWrapper


class Orca2(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."

        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{conversation}<|im_end|>\n<|im_start|>assistant"

        inputs = self.tokenizer(prompt, return_tensors='pt')
        output_ids = self.model.generate(inputs["input_ids"].to(self.model.device),)
        response = self.tokenizer.batch_decode(output_ids)[0]

        return response

model_core = Orca2
