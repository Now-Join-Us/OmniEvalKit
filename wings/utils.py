import os
import glob
import json
import torch
import ast
from tqdm import tqdm
from importlib import import_module


import transformers
import logging

import random
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_with_position_ids(q, k, cos, sin, position_ids_q, position_ids_image, unsqueeze_dim=1):
    cos_q = cos[position_ids_q].unsqueeze(unsqueeze_dim)
    sin_q = sin[position_ids_q].unsqueeze(unsqueeze_dim)
    cos_k = cos[position_ids_image].unsqueeze(unsqueeze_dim)
    sin_k = sin[position_ids_image].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

    return q_embed, k_embed

def log_args(model_args, data_args, training_args):
    def args2dict(args):
        return {k: str(v) for k, v in args.__dict__.items()}

    args_to_log = json.dumps(dict(
        model_args=args2dict(model_args),
        data_args=args2dict(data_args),
        training_args=args2dict(training_args)
    ), ensure_ascii=False, indent=2)
    if torch.cuda.current_device() == 0:
        logger.info(f"{args_to_log}")
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, 'model_data_training_args.json'),
        'w',
        encoding='utf-8'
    ) as f:
        f.write(args_to_log + '\n')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def import_class_from_string(full_class_string):
    module_path, _, class_name = full_class_string.rpartition('.')

    module = import_module(module_path)

    cls = getattr(module, class_name)
    return cls


def load_from_safetensors(model, model_safetensors_load_path):
    from safetensors import safe_open
    model_weights = {}
    file_path_list = [
        os.path.join(model_safetensors_load_path, 'model.safetensors')
    ] + glob.glob(
        os.path.join(model_safetensors_load_path, 'model-000*.safetensors')
    )
    if torch.cuda.current_device() == 0:
        logger.info('Loading from:\n' + '\n'.join(file_path_list) + '\n')
    for file_path in tqdm(file_path_list, desc='Loading safetensors shards: '):
        if os.path.exists(file_path):
            with safe_open(file_path, framework='pt') as f:
                for k in f.keys():
                    model_weights[k] = f.get_tensor(k)
    missing_keys = [name for name, _ in model.named_parameters() if name not in model_weights.keys()]
    more_keys = [name for name in model_weights.keys() if name not in model.state_dict().keys()]
    if torch.cuda.current_device() == 0:
        logger.info('Extra & Missing keys:\n' + f'Extra: {more_keys}\n' + f'Missing: {missing_keys}\n')
        if len(missing_keys) != 0:
            logger.warning(f'({len(missing_keys)}) missing_keys copied from source model.')
    for k in missing_keys:
        model_weights[k] = model.state_dict()[k]

    return model_weights


def set_gradient(model, vision_tower, training_args):
    if training_args.tuned_keys is not None:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if any([i_key in name for i_key in training_args.tuned_keys]):
                param.requires_grad_(True)

    if training_args.tune_mm_projector:
        for param in model.get_model().mm_projector.parameters():
            param.requires_grad_(True)
    else:
        for param in model.get_model().mm_projector.parameters():
            param.requires_grad_(False)

    if training_args.tune_llm:
        for name, param in model.named_parameters():
            if 'model.layers.' in name or name in ['model.norm.weight', 'lm_head.weight', 'model.embed_tokens.weight', 'model.tok_embeddings.weight']:
                param.requires_grad_(True)
    if training_args.tune_vision_tower:
        vision_tower.requires_grad_(True)
    else:
        vision_tower.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.tune_only_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        if torch.cuda.current_device() == 0:
            logger.warning('Only tune mm_mlp_adapter.')


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_only_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
