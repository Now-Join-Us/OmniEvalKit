import torch
from infer.base import InferCenter

class InferCoTCenter(InferCenter):
    """chain of thought inference

    Args:
        model_wrapper: object of model wrapper
    """
    def __init__(self, model_wrapper, **kwargs):
        super().__init__(model_wrapper)

    def infer(self, data, device=torch.device('cuda'), **kwargs):
        return self.infer_chain_of_thought(data, device)

infer_core = InferCoTCenter
