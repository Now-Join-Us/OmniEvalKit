from transformers import pipeline
from models.base import ModelWrapper

class ZephyrGemma(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_text_only(self, conversation, **kwargs):
        messages = [
            {
                "role": "system",
                "content": "",  # Model not yet trained for follow this
            },
            {"role": "user", "content": conversation},
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            stop_sequence="<|im_end|>",
        )
        response = outputs[0]["generated_text"][-1]["content"]
        return response

model_core = ZephyrGemma
