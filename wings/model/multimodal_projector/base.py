import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

import torch

class VisionProjector:
    def __init__(self, config, delay_load=False, device=torch.device('cuda'), **kwargs):
        self.config = config
        self.delay_load = delay_load
        self.kwargs = kwargs
        self.device = device

        self.projector = self._build_projector()

    def _build_projector(self):
        projector_type = getattr(self.config, 'mm_projector_type', 'linear')

        if projector_type == 'linear':
            return nn.Linear(self.config.mm_hidden_size, self.config.hidden_size).to(self.device)

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.extend([
                    nn.GELU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size)
                ])
            return nn.Sequential(*modules).to(self.device)

        if projector_type == 'identity':
            return IdentityMap().to(self.device)

        raise ValueError(f'Unknown projector type: {projector_type}')

    def to(self, device):
        self.device = device
        self.projector.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.projector(*args, **kwargs)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
