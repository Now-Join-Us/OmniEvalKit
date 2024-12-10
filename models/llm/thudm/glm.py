from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.base import ModelWrapper
import torch

class GLM(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_args)
        self.model = self.model.half().cuda()
        self.model.eval()

    def generate_text_only(self, conversation, **kwargs):
        inputs = self.tokenizer(conversation + ' [MASK]', return_tensors="pt")
        inputs = self.tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=1024, eos_token_id=self.tokenizer.eop_token_id)
        return self.tokenizer.decode(outputs[0].tolist())

model_core = GLM
