# Copyright (C) 2024 AIDC-AI
from utils import get_max_length
from configs import MODEL2MODULE
import importlib

from abc import abstractmethod

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import QuantizationMethod



class ModelWrapper(object):
    def __init__(self, model=None, tokenizer=None, model_path=None, model_args=None, tokenizer_args=None):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_path is not None and model_args is not None and tokenizer_args is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)

        if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
            self.max_length = get_max_length(self.model, self.tokenizer) if self.model is not None and self.tokenizer is not None else None
        self.force_use_generate = False

    def to(self, device):
        if hasattr(self, 'model') and not getattr(self.model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            try:
                self.model.to(device)
            except RuntimeError as e:
                pass
        return self

    def eval(self):
        if hasattr(self, 'model'):
            self.model.eval()
        return self

    def tie_weights(self):
        if hasattr(self, 'model') and hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
        return self

    def get_llm(self):
        if hasattr(self, 'model'):
            return self.model
        return None

    def get_tokenizer(self):
        if hasattr(self, 'tokenizer'):
            return self.tokenizer
        return None

    @abstractmethod
    def generate_text_only_from_token_id(self, conversation, **kwargs):
        raise NotImplementedError

    def is_overridden_generate_text_only_from_token_id(self, obj):
        return ModelWrapper.__dict__['generate_text_only_from_token_id'] is not obj.generate_text_only_from_token_id.__func__

    def _wrap_method(self, method):
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapper

    @abstractmethod
    def generate_text_only(self, conversation, **kwargs):
        raise NotImplementedError

    def is_overridden_generate_text_only(self, obj):
        return ModelWrapper.__dict__['generate_text_only'] is not obj.generate_text_only.__func__

    def generate_with_chat(self, tokenizer, conversation, history=[], **kwargs):
        response, _ = self.model.chat(tokenizer, conversation, history=history, **kwargs)
        return response

def get_general_model(model_path, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

def get_general_tokenizer(model_path, **kwargs):
    return AutoTokenizer.from_pretrained(model_path, **kwargs)

def get_model(model_name, model_path, model_args, tokenizer_args):
    model_module = MODEL2MODULE.get(model_name, MODEL2MODULE.get(model_name[:model_name.find('__')], None))

    if model_module is not None:
        module = importlib.import_module(f'models.{model_module}')
        model_core = getattr(module, 'model_core')
        model = model_core(model_path, model_args, tokenizer_args)
    else:
        model = get_general_model(model_path, **model_args)

    if not isinstance(model, ModelWrapper):
        try:
            tokenizer = getattr(module, 'tokenizer')
        except:
            tokenizer = get_general_tokenizer(model_path, **tokenizer_args)
        return ModelWrapper(model=model, tokenizer=tokenizer)
    return model
