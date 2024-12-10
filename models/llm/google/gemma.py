from models.base import ModelWrapper

# google/gemma-2b model_args:device_map=auto
#  google/gemma-2b-it   model_args:device_map="auto",torch_dtype=torch.bfloat16
#  google/gemma-7b-it   model_args:device_map="auto",torch_dtype=torch.bfloat16
#  google/gemma-2-9b-it   model_args:device_map="auto",torch_dtype=torch.bfloat16

class Gemma(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        input_ids = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**input_ids)
        return self.tokenizer.decode(outputs[0])

model_core = Gemma
