from models.base import ModelWrapper

class QwenCoder2(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer(conversation, return_tensors='pt').to(self.model.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response

    def generate_k_tokens(self, conversation, **gen_kwargs):
        # import pdb; pdb.set_trace()
        inputs = self.tokenizer(conversation, return_tensors='pt').to(self.model.device)
        generated_tokens = self.model.generate(
                            **inputs,
                            # num_return_sequences=gen_kwargs.get('num_return_sequences', 1),
                            **gen_kwargs,
                        )
        return generated_tokens 
model_core = QwenCoder2
