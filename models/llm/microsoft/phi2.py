from models.base import ModelWrapper

class Phi2(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer(conversation, return_tensors="pt", return_attention_mask=False)

        outputs = self.model.generate(**inputs, max_length=200)
        response = self.tokenizer.batch_decode(outputs)[0]
        return response

model_core = Phi2
