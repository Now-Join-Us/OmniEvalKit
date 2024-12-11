from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper

# CUDA_VISIBLE_DEVICES="0" python main.py --data  --model MobileLLM-125M --model_args use_fast:False --tokenizer_args trust_remote_code:True --time_str 03_26_00_00_00

class MobileLLM(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__(model_path=model_path, model_args=model_args, tokenizer_args=tokenizer_args)

    def generate_text_only(self, conversation, **kwargs):
        raise NotImplementedError

model_core = MobileLLM
