from models.base import ModelWrapper

class Phi3(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": conversation},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        response = output[0]['generated_text']

        return response

model_core = Phi3
