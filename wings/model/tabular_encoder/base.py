import os
from wings.model.tabular_encoder.tp_berta import TPBERTa

def build_tabular_encoder(model_args, data_args, **kwargs):
    return TPBERTa(
        data_path=data_args.tabular_source_data_path,
        dataset_list=data_args.tabular_source_data.split(','),
        pretrain_dir=model_args.tabular_encoder_path,
        cache_path=model_args.tabular_cache_path
    )
