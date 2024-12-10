# Copyright (C) 2024 AIDC-AI
import os
import torch
import time
from tqdm import tqdm

from models.base import get_model
from evals.base import InferCenter, EvalTool
from dataloaders.utils import get_data
from utils import setup_args, Response, save_json, get_rank_and_world_size, rank_zero_check, calculate_model_flops, batchify


def get_model_dataset_to_inference(model, data, log_path, rank, world_size, disable_infer=False):
    model_data_not_inferred, model_data_is_inferred = [], []
    for model_name, model_path in model.items():
        for dataset_name, dataset_file_path in tqdm(data.items()):
            model_data_pair = ((model_name, model_path), (dataset_name, dataset_file_path))
            resps = Response(os.path.join(log_path, model_name, dataset_name), rank=rank, world_size=world_size)
            dataset = get_data(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, preloaded_image_num=0)
            is_inferred = True
            for idx, data in enumerate(dataset):
                if data['id'] not in resps.keys():
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
    model_data_not_inferred, model_data_is_inferred = get_model_dataset_to_inference(args.model, args.data, args.log_path, rank, world_size, args.disable_infer)

    for (model_name, model_path), (dataset_name, dataset_file_path) in tqdm(model_data_not_inferred):
        print(model_name, dataset_name, 'infer')
        model_wrapper = get_model(model_name, model_path, args.model_args, args.tokenizer_args)
        model_wrapper.to(device).eval().tie_weights()

        log_path = os.path.join(args.log_path, model_name, dataset_name)

        resps = Response(log_path, save_steps=args.save_steps, rank=rank, world_size=world_size)
        dataset = get_data(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=args.image_url, preloaded_image_num=args.preloaded_image_num)

        if args.set_calculate_flops:
            calculate_model_flops(model_wrapper, model_name)

        for idx, data_idx in enumerate(tqdm(range(0, len(dataset), args.batch_size))):
            data = batchify(dataset, data_idx, args.batch_size)
            if isinstance(data, dict) and data['id'] in resps.keys():
                continue

            center = InferCenter(model_wrapper)
            resp = center.infer(data=data)

            if isinstance(data, list):
                resps.update({i_data['id']: resp[i_data['id']] for i_data in data})
            else:
                resps.update({data['id']: resp})
        resps.save()
        model_data_is_inferred.append(((model_name, model_path), (dataset_name, dataset_file_path)))

    filter_model_wrapper = None
    if 'model' in args.filter_type:
        assert args.filter_model is not None
        filter_model_wrapper = get_model(
            model_name=args.filter_model.split('/')[-1],
            model_path=args.filter_model,
            model_args=args.filter_model_args,
            tokenizer_args=args.filter_tokenizer_args
        )
        filter_model_wrapper.to(device).eval().tie_weights()

    for (model_name, model_path), (dataset_name, dataset_file_path) in tqdm(model_data_is_inferred):
        print(model_name, dataset_name, 'eval')
        dataset = get_data(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, preloaded_image_num=0)
        log_path = os.path.join(args.log_path, model_name, dataset_name)
        resps = Response(log_path, save_steps=args.save_steps, rank=rank, world_size=world_size)
        scored_dataset_file_path, statistics_path = os.path.join(log_path, f'scored_dataset.json'), os.path.join(log_path, f'statistics.json')

        tool = EvalTool(
            dataset_name=dataset_name,
            dataset=dataset,
            filter_type=args.filter_type,
            filter_model_wrapper=filter_model_wrapper
        )
        statistics = tool.evaluate(resps, scored_dataset_file_path, statistics_path)
        print(statistics)
