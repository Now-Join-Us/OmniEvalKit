import torch
import transformers

from dataclasses import dataclass, field

from typing import Optional, List


@dataclass
class DataArguments:
    data_name: str # a|b|c
    data_processor: Optional[str] = field(default=None)
    data_mode: Optional[str] = field(default=None)
    processed_data_dir: Optional[str] = field(default=None)
    is_multimodal: bool = field(default=True)
    image_aspect_ratio: str = field(default='pad')
    only_image_data: bool = False
    image_text_data_ratio: float = 0.
    image_token_length: Optional[int] = field(default=729)

    data: str = ''
    model: str = ''
    work_dir: str = '.'
    mode: str = 'all'
    nproc: int = 4
    retry: int = None
    ignore: bool = False
    verbose: bool = False
    prefetch: bool = False
    time_str: str = ''

@dataclass
class ModelArguments:
    model_name: str
    model_path: str
    vision_tower: str
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_projector_type: Optional[str] = field(default='linear')
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    system_prompt_length: Optional[int] = field(default=14)

    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    model_safetensors_load_path: Optional[str] = None

    moe_tune_mm_projector: bool = True
    v_enable: bool = False
    dmoe_enable: bool = False
    dmoe_tune_mm_projector: bool = True
    dmoe_params_init_mode: str = 'copy'
    dmoe_mode: str = ''
    dmoe_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place moe layers."})
    dlmoe_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place dlmoe layers."})
    dlatt_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place dlatt layers."})

    damoe_enable: bool = False
    dlmoe_enable: bool = False
    dlatt_enable: bool = False
    dlora_enable: bool = False
    damoe_ep_size: int = 1
    damoe_top_k_experts: int = 2
    damoe_capacity_factor: float = 1.
    damoe_eval_capacity_factor: float = 2.
    damoe_min_capacity: int = 0
    damoe_use_residual: bool = False
    damoe_router_aux_loss_coef: float = 0.01
    peft_enable: bool = False
    peft_mode: str = ''
    peft_kwargs_k: Optional[List[str]] = field(default_factory=list)
    peft_kwargs_v: Optional[List[str]] = field(default_factory=list)

    lora_enable: bool = False
    lora_r: bool = False
    lora_dim: int = 8
    attn_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place attn layers."})

    wings_router_type: str = 'linear'
    moe_enable: bool = False
    train_modules: Optional[List[str]] = field(default=None, metadata={"help": ""})
    moe_mode: str = field(
        default="second_half",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["first_half", "second_half", "sparse", "dense", "debug"],
        },
    )
    ep_size: int = 1
    num_experts: Optional[List[int]] = field(default=4, metadata={"help": "number of experts for each moe layer."})
    top_k_experts: int = field(
        default=2,
        metadata={
            "help": "Top-k experts to deal with tokens.",
            "choices": [1, 2],
        },
    )
    capacity_factor: float = 1.
    eval_capacity_factor: float = 2.
    min_capacity: int = 0
    use_residual: bool = False
    router_aux_loss_coef: float = 0.01

    system_slot: str = "<|im_start|>system\n"
    user_slot: str =  "<|im_start|>user\n"
    gpt_slot: str = "<|im_start|>assistant\n"
    eot: str = "<|im_end|>"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    vision_tower_lr_follow_mm_projector: Optional[bool] = False
    lr_projector_follow_tuned_keys: Optional[List[str]] = field(default=None, metadata={"help": "where to place moe layers."})
    group_by_modality_length: bool = field(default=False)
    use_cache: bool = field(default=False)
    tuned_keys: Optional[List[str]] = field(default=None)
    tune_mm_projector: bool = field(default=False)
    tune_llm: bool = field(default=False)
    tune_vision_tower: bool = field(default=False)
    tune_only_mm_mlp_adapter: bool = field(default=False)


    def unfreeze(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True
