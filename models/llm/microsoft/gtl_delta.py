from models.base import ModelWrapper

class GTL_DELTA(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversations, device, **kwargs):
        print(device)
        inputs = self.tokenizer(conversations, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            max_new_tokens=512
        )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:])

model_core = GTL_DELTA
