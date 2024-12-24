# Copyright (C) 2024 AIDC-AI
import os
import torch
import time
import importlib
from tqdm import tqdm

from evals.base import EvalTool
from dataloaders.utils import get_data
from utils import setup_args, Response, save_json, get_rank_and_world_size, rank_zero_check, calculate_model_flops, batchify, get_log_path, logger

def get_model_dataset_to_inference(model, data, log_path, infer_type, rank, world_size, disable_infer=False):
    model_data_not_inferred, model_data_is_inferred = [], []
    for model_name, model_path in model.items():
        for dataset_name, dataset_file_path in tqdm(data.items()):
            model_data_pair = ((model_name, model_path), (dataset_name, dataset_file_path))
            responses = Response(get_log_path(log_path, infer_type, model_name, dataset_name), rank=rank, world_size=world_size)
            dataset = get_data(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, preloaded_image_num=0)
            is_inferred = True
            for idx, data in enumerate(dataset):
                if data['id'] not in responses.keys():
                    model_data_not_inferred.append(model_data_pair)
                    is_inferred = False
                    break
            if is_inferred:
                model_data_is_inferred.append(model_data_pair)
    if disable_infer:
        return [], model_data_is_inferred
    return model_data_not_inferred, model_data_is_inferred

if __name__ == "__main__":
    args = setup_args()
    device = torch.device('cuda')

    rank, world_size = get_rank_and_world_size()
    model_data_not_inferred, model_data_is_inferred = get_model_dataset_to_inference(args.model, args.data, args.log_path, args.infer_type, rank, world_size, args.disable_infer)

    if not args.only_do_eval:
        from models.base import get_model
        for (model_name, model_path), (dataset_name, dataset_file_path) in tqdm(model_data_not_inferred):
            logger.info(f'[INFER] {model_name} on {dataset_name}')
            model_wrapper = get_model(model_name, model_path, args.model_args, args.tokenizer_args)
            model_wrapper.to(device).eval().tie_weights()

            log_path = get_log_path(args.log_path, args.infer_type, model_name, dataset_name)

            responses = Response(log_path, save_steps=args.save_steps, rank=rank, world_size=world_size)
            dataset = get_data(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=args.image_url, preloaded_image_num=args.preloaded_image_num)

            if args.set_calculate_flops:
                calculate_model_flops(model_wrapper, model_name)

            for idx, data_idx in enumerate(tqdm(range(0, len(dataset), args.batch_size))):
                data = batchify(dataset, data_idx, args.batch_size)
                if isinstance(data, dict) and data['id'] in responses.keys():
                    continue

                infer_module = importlib.import_module(f'infer.{args.infer_type}') # 等价于 from infer.chain_of_thought
                center = getattr(infer_module, 'infer_core')(model_wrapper, **args.infer_args) # 等价于 from infer.chain_of_thought import infer_core as center
                resp = center.infer(data=data, **args.infer_args)
                if isinstance(data, list):
                    responses.update({i_data['id']: resp[i_data['id']] for i_data in data})
                else:
                    responses.update({data['id']: resp})
            responses.save()
            model_data_is_inferred.append(((model_name, model_path), (dataset_name, dataset_file_path)))

    filter_model_wrapper = None
    if 'model_based' in args.filter_type:
        assert args.filter_model is not None
        filter_model_wrapper = get_model(
            model_name=args.filter_model.split('/')[-1],
            model_path=args.filter_model,
            model_args=args.filter_model_args,
            tokenizer_args=args.filter_tokenizer_args
        )
        filter_model_wrapper.to(device).eval().tie_weights()

    for (model_name, model_path), (dataset_name, dataset_file_path) in tqdm(model_data_is_inferred):
        logger.info(f'[EVAL] {model_name} on {dataset_name}')
        dataset = get_data(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, preloaded_image_num=0)
        log_path = get_log_path(args.log_path, args.infer_type, model_name, dataset_name)
        responses = Response(log_path, save_steps=args.save_steps, rank=rank, world_size=world_size)
        scored_dataset_file_path, statistics_path = os.path.join(log_path, f'scored_dataset.json'), os.path.join(log_path, f'statistics.json')

        tool = EvalTool(
            dataset_name=dataset_name,
            dataset=dataset,
            filter_type=args.filter_type,
            filter_model_wrapper=filter_model_wrapper,
            **args.eval_args
        )
        statistics = tool.evaluate(responses, scored_dataset_file_path, statistics_path)
        logger.info(statistics)
