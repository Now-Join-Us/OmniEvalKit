# Copyright (C) 2024 AIDC-AI
import gc
from typing import Union

import torch


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

