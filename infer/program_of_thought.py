# This file includes functions adapted from the Program-of-Thoughts repository (https://github.com/TIGER-AI-Lab/Program-of-Thoughts).
# Original work by chen et al., licensed under MIT license.
# Copyright (c) 2024 wenhu chen

import torch
from infer.base import InferCenter

class InferPoTCenter(InferCenter):
    """program of thought inference

    Args:
        model_wrapper: object of model wrapper
    """
    def __init__(self, model_wrapper, **kwargs):
        super().__init__(model_wrapper)

    def infer(self, data, device=torch.device('cuda'), **kwargs):
        return self.infer_program_of_thought(data, device)

infer_core = InferPoTCenter
