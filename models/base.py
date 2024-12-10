# Copyright (C) 2024 AIDC-AI
from models.utils import stop_sequences_criteria
from utils import get_max_length
from configs import MODEL2MODULE
import importlib

from typing import List, Tuple
from abc import abstractmethod
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import QuantizationMethod

from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)

def tok_encode(
    tokenizer, string: str, left_truncate_len=None, add_special_tokens=None, add_bos_token=False
) -> List[int]:
    """ """
    # default for None - empty dict, use predefined tokenizer param
    # used for all models except for CausalLM or predefined value
    special_tokens_kwargs = {}

    # by default for CausalLM - false or self.add_bos_token is set
    if add_special_tokens is None:
        special_tokens_kwargs = {
            "add_special_tokens": False or add_bos_token
        }
    # otherwise the method explicitly defines the value
    else:
        special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

    encoding = tokenizer.encode(string, **special_tokens_kwargs)

    # left-truncate the encoded context to be at most `left_truncate_len` tokens long
    if left_truncate_len:
        encoding = encoding[-left_truncate_len:]

    return encoding

def tok_decode(tokenizer, tokens, skip_special_tokens=True):
    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

def tok_batch_encode(
    tokenizer,
    strings: List[str],
    padding_side: str = "left",
    left_truncate_len: int = None,
    truncation: bool = False,
    add_bos_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    add_special_tokens = {"add_special_tokens": False or add_bos_token}

    encoding = tokenizer(
        strings,
        truncation=truncation,
        padding="longest",
        return_tensors="pt",
        **add_special_tokens,
    )
    if left_truncate_len:
        encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
        encoding["attention_mask"] = encoding["attention_mask"][
            :, -left_truncate_len:
        ]
    tokenizer.padding_side = old_padding_side

    return encoding["input_ids"], encoding["attention_mask"]

def _model_generate(model, tokenizer, context, max_length, stop, **generation_kwargs):
    # temperature = 0.0 if not set
    # if do_sample is false and temp==0.0:
    # remove temperature, as do_sample=False takes care of this
    # and we don't want a warning from HF
    generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
    do_sample = generation_kwargs.get("do_sample", None)

    # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
    if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
        generation_kwargs["do_sample"] = do_sample = False

    if do_sample is False and generation_kwargs.get("temperature") == 0.0:
        generation_kwargs.pop("temperature")
    # build stopping criteria
    stopping_criteria = stop_sequences_criteria(
        tokenizer, stop, context.shape[1], context.shape[0]
    )

    return model.generate(
        input_ids=context,
        max_length=max_length,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        **generation_kwargs,
    )

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

    def accelerate(self):
        # gpus = torch.cuda.device_count()
        # accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        # accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        # if accelerator.num_processes > 1:
        #     self.accelerator = accelerator

        # self._device = torch.device(device)

        # model_kwargs.update(
        #     _get_accelerate_args(
        #         device_map_option,  # TODO: phase out device_map_option?
        #         max_memory_per_gpu,
        #         max_cpu_memory,
        #         offload_folder,
        #         gpus,
        #     )
        # )
        pass

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
