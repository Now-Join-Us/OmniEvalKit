# Copyright (C) 2024 AIDC-AI
from configs import _DEFAULT_MAX_LENGTH, MODEL_PATH, DATA_PATH, DATASET2FILE, OUTPUT_PATH
from typing import List, Any, Dict
import transformers
import json
import re
import os
import numpy as np
import argparse
import pickle
import torch
from datetime import datetime
from pathlib import Path
from pprint import pprint
from PIL import Image
from collections import Counter
from collections.abc import Iterable


import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

def get_max_length(model, tokenizer, _max_length=None):
    if _max_length:  # if max length manually set, return it
        return _max_length
    seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
    for attr in seqlen_config_attrs:
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length == 1000000000000000019884624838656:
            return _DEFAULT_MAX_LENGTH
        return tokenizer.model_max_length
    return _DEFAULT_MAX_LENGTH

def handle_arg_string(arg):
    if isinstance(arg, str):
        if arg.lower() == "true":
            return True
        elif arg.lower() == "false":
            return False
        elif arg.startswith('torch.'):
            torch_attr = arg.split('.')[1]
            return getattr(torch, torch_attr)
    elif arg.isnumeric():
        return int(arg)
    try:
        return int(arg)
    except ValueError:
        pass
    try:
        return float(arg)
    except ValueError:
        return arg

def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)
dataset_list = ['aclue', 'anli', 'arc_challenge_25', 'arc_easy_25', 'boolq', 'cb', 'cmmlu', 'cola', 'glue', 'mmlu_5', 'mnli', 'mrpc']

from jinja2 import Environment, BaseLoader, StrictUndefined
env = Environment(loader=BaseLoader, undefined=StrictUndefined)
env.filters["regex_replace"] = regex_replace

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue


from tqdm import tqdm

def batchify(dataset, data_idx, batch_size):
    if batch_size == 1:
        return dataset[data_idx]
    return [dataset[i] for i in range(data_idx, min(data_idx + batch_size, len(dataset)))]

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", "-d", type=str, default=None, help="Datasets (in lower, split by `,`) e.g. `mmlu_5`"
    )
    parser.add_argument(
        "--time_str", "-t", type=str, default="", help="Timestamp e.g. `05_30_06_13_26`"
    )

    parser.add_argument(
        "--infer_type",
        type=str,
        default="direct",
        choices=[
            'direct',
            'chain_of_thought',
            'tree_of_thought',
            'program_of_thought',
            'self_consistency',
            'tool_integrated_reasoning',
            'coder'
        ],
        help="Sets the inference method, various inferring (reasoning) techniques such as `direct`, `chain_of_thought`, `tree_of_thought`, `program_of_thought`, `self_consistency`, and `tool_integrated_reasoning`."
    )
    parser.add_argument(
        "--infer_args",
        type=str,
        default="",
        help="Comma separated string arguments for inferring, e.g. `device_map=auto`",
    )

    parser.add_argument(
        "--model", "-m", type=str, default="Qwen/Qwen1.5-7B", help="Models e.g. `Qwen/Qwen1.5-7B`"
    )
    parser.add_argument(
        "--model_args",
        "-ma",
        type=str,
        default="",
        help="Comma separated string arguments for model, e.g. `device_map=auto`",
    )
    parser.add_argument(
        "--tokenizer_args",
        "-ta",
        type=str,
        default="",
        help="Comma separated string arguments for tokenizer, e.g. `use_fast=False`",
    )

    parser.add_argument(
        "--log_path",
        "-lp",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="How often to save intermediate results",
    )
    parser.add_argument(
        "--preloaded_image_num",
        type=int,
        default=1,
        help="The number of images pre-loaded in `__getitem__`.",
    )
    parser.add_argument(
        "--disable_infer",
        action="store_true",
        help="Flag to disable infer. Default is False.",
    )
    parser.add_argument(
        "--set_calculate_flops",
        action="store_true",
        help="Flag to calculate FLOPs. Default is False.",
    )

    parser.add_argument(
        "--data_url", type=str, default=None
    )
    parser.add_argument('--only_do_eval', action='store_true')
    parser.add_argument(
        "--filter_type", type=str, default="regex", choices=['regex', 'model_based', 'regex,model_based', 'extract_tail']
    )
    parser.add_argument(
        "--filter_args", type=str, default=""
    )
    parser.add_argument(
        "--filter_model", type=str, default=None, help="filter models e.g. `Qwen/Qwen1.5-7B`"
    )
    parser.add_argument(
        "--filter_model_args",
        default="",
        type=str,
        help="Comma separated string arguments for filter model, e.g. `device_map=auto`",
    )
    parser.add_argument(
        "--filter_tokenizer_args",
        default="",
        type=str,
        help="Comma separated string arguments for filter model tokenizer, e.g. `use_fast=False`",
    )
    parser.add_argument(
        "--eval_args", type=str, default=""
    ) # question_type=xxx,request_type=xxx,calculate_type=xxx,estimate_type=each_then_overall (or overall)
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch Size (Only used for retrieval)",
    )

    check_argument_types(parser)
    args = parser.parse_args()

    args_to_parse = [
        'infer_args',
        'model_args',
        'tokenizer_args',
        'filter_args',
        'filter_model_args',
        'filter_tokenizer_args',
        'eval_args'
    ]

    for arg_to_parse_name in args_to_parse:
        setattr(args, arg_to_parse_name, simple_parse_args_string(getattr(args, arg_to_parse_name)))

    if args.time_str == '':
        args.time_str = datetime.now().strftime('%m_%d_%H_%M_%S')

    model_dict = {}
    for model_path in args.model.split(','):
        if os.path.isdir(model_path):
            model_dict[os.path.basename(model_path)] = model_path
        else:
            model_dict[model_path] = os.path.join(MODEL_PATH, model_path)
    args.model = model_dict

    if args.filter_model is not None:
        args.filter_model = os.path.join(MODEL_PATH, args.filter_model)

    if args.data is None:
        assert args.data_url is not None
        args.data = {os.path.basename(file_name).replace('.json', ''): os.path.join(args.data_url, file_name) for file_name in os.listdir(args.data_url)}
    else:
        args.data = {
            os.path.basename(data_name).replace('.json', ''): os.path.join(
                DATA_PATH if args.data_url is None else args.data_url, DATASET2FILE.get(data_name, f'{data_name}.json')
            )
            for data_name in args.data.split(',')
        }
    args.image_url = os.path.join(DATA_PATH if args.data_url is None else args.data_url, 'images')
    if args.log_path is None:
        args.log_path = os.path.join(OUTPUT_PATH, args.time_str)

    pprint(vars(args))
    return args

class Response(Dict[str, Any]):
    def __init__(self, log_dir: str, save_steps: int = 5000, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.log_dir: str = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.pkl_file_path: str = os.path.join(self.log_dir, f'resps_tmp_{rank}_{world_size}.pkl')
        self.json_file_path: str = os.path.join(self.log_dir, f'resps_tmp_{rank}_{world_size}.json')
        self.save_steps: int = save_steps
        self.load()

    def load(self) -> None:
        if os.path.isfile(self.pkl_file_path):
            self.update(load_pickle(self.pkl_file_path))

    def save(self) -> None:
        if len(self) != 0:
            save_pickle(self.pkl_file_path, dict(self))
            save_json(self.json_file_path, dict(self))

    def update(self, resp: Dict[str, Any]) -> None:
        super().update(resp)
        if len(self) % self.save_steps == 0:
            self.save()

    def __len__(self) -> int:
        return super().__len__()

    def keys(self) -> Any:
        return super().keys()

def get_rank_and_world_size():
    local_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return local_rank, world_size

def rank_zero_check(rank, world_size, file_name_to_format):
    with open(file_name_to_format.format(rank, world_size), 'w', encoding='utf-8') as f:
        f.write('done\n')
    is_done = True
    if rank == 0:
        for i_rank in range(world_size):
            done_file_name = file_name_to_format.format(i_rank, world_size)
            if not os.path.exists(done_file_name):
                is_done = False
                print(f'{done_file_name} F Rank {i_rank} ..')
            else:
                print(f'{done_file_name} OK {i_rank} ..')
    return is_done

def calculate_model_flops(model_wrapper, model_name):
    from calflops import calculate_flops
    if not hasattr(model_wrapper, 'max_length'):
        model_wrapper.max_length = get_max_length(model_wrapper.model, model_wrapper.tokenizer)

    flops, macs, params = calculate_flops(model=model_wrapper.model,
                                          input_shape=(1, model_wrapper.max_length),
                                          transformer_tokenizer=model_wrapper.tokenizer)

    print(f"MODEL:{model_name}   FLOPs:{flops}   MACs:{macs}   Params:{params}")

def load_image(image_file_path):
    return Image.open(image_file_path).convert('RGB')

def get_log_path(base_path, infer_type, model_name, dataset_name):
    return os.path.join(base_path, model_name, dataset_name if infer_type == 'direct' else f'{dataset_name}_{infer_type}')

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, str) or not isinstance(item, Iterable):
            flattened.append(item)
        else:
            flattened.extend(flatten_list(item))
    return flattened

def most_common_length_strings(strings):
    ## 计算出现频率最高的字符串长度，并返回该长度的第一个字符串 ##
    lengths = [len(s) for s in strings]
    length_count = Counter(lengths)
    most_common_length = length_count.most_common(1)[0][0]
    result = [s for s in strings if len(s) == most_common_length]
    return result[0]
