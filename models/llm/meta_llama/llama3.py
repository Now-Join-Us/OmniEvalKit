from models.base import ModelWrapper
from transformers import pipeline

# meta-llama/Meta-Llama-3-8B  model_args: model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
class LLaMA3Base(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs=model_args
        )

    def generate_text_only(self, conversation, **kwargs):
        outputs = pipeline(conversation, max_new_tokens=512)
        return outputs[0]["generated_text"]

model_core = LLaMA3Base
