import transformers
from models.base import ModelWrapper

class RedPajama(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        MIN_TRANSFORMERS_VERSION = '4.25.1'
        # check transformers version
        assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        prompt = f"<human>: {conversation}\n<bot>:"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True, pad_token_id=self.tokenizer.eos_token_id
        )
        token = outputs.sequences[0, input_length:]
        response = self.tokenizer.decode(token)
        return response

model_core = RedPajama
