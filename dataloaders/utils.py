import re
import importlib

from configs import DATASET2MODULE
from dataloaders.base import Dataset

def get_dataset(dataset_name, **kwargs):
    if dataset_name in DATASET2MODULE.keys():
        module = importlib.import_module(f'dataloaders.{DATASET2MODULE[dataset_name]}') # means `from dataloaders.bbh`
        data_core = getattr(module, 'data_core') # means `from dataloaders.bbh import data_core`
        dataset = data_core(dataset_name=dataset_name, **kwargs)
    else:
        dataset = Dataset(dataset_name=dataset_name, **kwargs)

    return dataset
