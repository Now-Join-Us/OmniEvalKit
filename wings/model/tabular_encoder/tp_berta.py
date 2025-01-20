
import os
import numpy as np
import dataclasses as dc
import typing as ty
from pathlib import Path
from tqdm import tqdm
import torch
from copy import deepcopy
from sklearn.impute import SimpleImputer

from wings.model.tabular_encoder.tp_berta_lib import DataConfig, prepare_tpberta_loaders
from wings.model.tabular_encoder.tp_berta_lib.modeling import build_default_model
from wings.utils import load_json


def data_nan_process(N_data, C_data, num_nan_policy='mean', cat_nan_policy='new', num_new_value=None, imputer=None, cat_new_value=None):
    """
    Process the NaN values in the dataset.

    :param N_data: ArrayDict
    :param C_data: ArrayDict
    :param num_nan_policy: str
    :param cat_nan_policy: str
    :param num_new_value: Optional[np.ndarray]
    :param imputer: Optional[SimpleImputer]
    :param cat_new_value: Optional[str]
    :return: Tuple[ArrayDict, ArrayDict, Optional[np.ndarray], Optional[SimpleImputer], Optional[str]]
    """
    if N_data is None:
        N = None
    else:
        N = deepcopy(N_data)
        if 'train' in N_data.keys():
            if N['train'].ndim == 1:
                N = {k: v.reshape(-1, 1) for k, v in N.items()}
        else:
            if N['test'].ndim == 1:
                N = {k: v.reshape(-1, 1) for k, v in N.items()}
        N = {k: v.astype(float) for k, v in N.items()}
        num_nan_masks = {k: np.isnan(v) for k, v in N.items()}
        if any(x.any() for x in num_nan_masks.values()):
            if num_new_value is None:
                if num_nan_policy == 'mean':
                    num_new_value = np.nanmean(N_data['train'], axis=0)
                elif num_nan_policy == 'median':
                    num_new_value = np.nanmedian(N_data['train'], axis=0)
                else:
                    raise ValueError(f'Unknown numerical NaN policy: {num_nan_policy}')
            for k, v in N.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_value, num_nan_indices[1])
    if C_data is None:
        C = None
    else:
        assert(cat_nan_policy == 'new')
        C = deepcopy(C_data)
        if 'train' in C_data.keys():
            if C['train'].ndim == 1:
                C = {k: v.reshape(-1, 1) for k, v in C.items()}
        else:
            if C['test'].ndim == 1:
                C = {k: v.reshape(-1, 1) for k, v in C.items()}
        C = {k: v.astype(str) for k, v in C.items()}
        # assume the cat nan condition
        cat_nan_masks = {k: np.isnan(v) if np.issubdtype(v.dtype, np.number) else np.isin(
            v, ['nan', 'NaN', '', None]) for k, v in C.items()}
        if any(x.any() for x in cat_nan_masks.values()):
            if cat_nan_policy == 'new':
                if cat_new_value is None:
                    cat_new_value = '___null___'
                    imputer = None
            elif cat_nan_policy == 'most_frequent':
                if imputer is None:
                    cat_new_value = None
                    imputer = SimpleImputer(strategy='most_frequent')
                    imputer.fit(C['train'])
            else:
                raise ValueError(f'Unknown categorical NaN policy: {cat_nan_policy}')
            if imputer:
                C = {k: imputer.transform(v) for k, v in C.items()}
            else:
                for k, v in C.items():
                    cat_nan_indices = np.where(cat_nan_masks[k])
                    v[cat_nan_indices] = cat_new_value

    result = (N, C, num_new_value, imputer, cat_new_value)
    return result

MASK_TOKEN_ID = 50264 # RoBerta Tokenizer [MASK] ID

ArrayDict = ty.Dict[str, np.ndarray]
REGRESSION = 'regression'
CLASSIFICATION = 'classification'


def remove_htags(s, sub_s):
    if s.startswith(sub_s):
        s = s[len(sub_s):]
    
    if s.endswith(sub_s):
        s = s[:-len(sub_s)]
    
    return s

def parse_table(table_str):
    lines = remove_htags(table_str, '<h>').split('<h>')
    header = remove_htags(lines[0], '<g>').split('<g>')
    data_rows = [
        [element.replace('[MASK]', '<mask>') for element in remove_htags(line, '<g>').split('<g>')]
        for line in lines[1:]
    ]
    return header, data_rows

def extract_columns(header, data_rows, col_names):
    col_name_to_index = {col_name: header.index(col_name) if col_name in header else -1 for col_name in col_names}
    target_index_to_source = {target_col_idx: header.index(col_name) if col_name in header else -1 for target_col_idx, col_name in enumerate(col_names)}

    new_rows = []
    for row in data_rows:
        new_row = [
            row[col_name_to_index[col_name]] if col_name_to_index[col_name] != -1 else '[ERROR]'
            for col_name in col_names
        ]
        new_rows.append(new_row)

    result_array = np.array(new_rows, dtype=object)
    return result_array, target_index_to_source

def reorder_tensor(tensor, index_order):
    sorted_indices = [k for k, v in sorted(index_order.items(), key=lambda item: item[1]) if v != -1 and k >= 0 and k < tensor.shape[1]]
    return tensor[:, sorted_indices, :]

@dc.dataclass
class TabularDataset:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    meta_data: ty.Dict[str, ty.Any]
    folder: ty.Optional[Path]

    @classmethod
    def from_dir(cls, dir_: ty.Union[Path, str]):
        dir_ = Path(dir_)

        def load(item) -> ArrayDict:
            return {
                # type: ignore[code]
                x: ty.cast(np.ndarray, np.load(
                    dir_ / f'{item}_{x}.npy', allow_pickle=True))
                for x in ['train', 'val', 'test']
            }

        def preprocess(N, C, y, meta_data):
            # clean the feature name
            for i in range(len(meta_data['feature_intro']['num']['name'])):
                # remove the period in the feature name
                if meta_data['feature_intro']['num']['name'][i].endswith('.'):
                    meta_data['feature_intro']['num']['name'][i] = meta_data['feature_intro']['num']['name'][i][:-1]
            for i in range(len(meta_data['feature_intro']['cat']['name'])):
                # remove the period in the feature name
                if meta_data['feature_intro']['cat']['name'][i].endswith('.'):
                    meta_data['feature_intro']['cat']['name'][i] = meta_data['feature_intro']['cat']['name'][i][:-1]
            if isinstance(meta_data['target_intro'], list):
                meta_data['target_intro'] = meta_data['target_intro'][0]
            meta_data['target_intro'] = meta_data['target_intro'].replace(
                '.', '')
            if isinstance(meta_data['task_intro'], list):
                meta_data['task_intro'] = meta_data['task_intro'][0]
            # change the type of N if it is not None and the type is not float
            # if C is not None and C['train'].dtype.kind in ['U', 'S']:
            #     for key in C.keys():
            #         C[key] = C[key].astype(str)
            if C is not None:
                for key in C.keys():
                    C[key] = C[key].astype(str)
            if N is not None and not isinstance(N['train'], float):
                values_to_replace = [' ', '?', 'nan']
                for key in N.keys():
                    
                    if N[key].dtype.kind in ['U', 'S']:
                        N[key] = N[key].astype(str)
                        N[key] = N[key].astype(object)
                    # print(N[key])
                    
                    N[key] = np.where(
                        np.isin(N[key], values_to_replace), np.nan, N[key])
                    N[key] = N[key].astype(float)
            if y['train'].dtype.kind in ['U', 'S']:
                for key in y.keys():
                    y[key] = y[key].astype(str)
            if meta_data['task_type'] == CLASSIFICATION:
                y = {key: y[key].astype(str) for key in y.keys()}
            elif meta_data['task_type'] == REGRESSION:
                y = {key: y[key].astype(float) for key in y.keys()}
            if len(y['train'].shape) > 1:
                y = {key: y[key].squeeze() for key in y.keys()}
            N, C, num_new_value, imputer, cat_new_value = data_nan_process(
                N, C)

            return N, C, y, meta_data
        N = load('N') if dir_.joinpath('N_train.npy').exists() else None

        C = load('C') if dir_.joinpath('C_train.npy').exists() else None
        y = load('y')
        meta_data = load_json(dir_ / 'meta_data.json')
        N, C, y, meta_data = preprocess(N, C, y, meta_data)
        return TabularDataset(
            N,
            C,
            y,
            meta_data,
            dir_,
        )

    @property
    def is_classification(self) -> bool:
        return self.meta_data['task_type'] == CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        return self.meta_data['task_type'] == REGRESSION

    @property
    def num_features_name(self) -> ty.List[str]:
        return [i.strip() for i in self.meta_data['feature_intro']['num']['name']]

    @property
    def cat_features_name(self) -> ty.List[str]:
        return [i.strip() for i in self.meta_data['feature_intro']['cat']['name']]

    @property
    def feature_name(self) -> ty.List[str]:
        return self.num_features_name+self.cat_features_name

    @property
    def label_name(self) -> str:
        return self.meta_data['target_intro']

    @property
    def n_num_features(self) -> int:
        return len(self.num_features_name)

    @property
    def n_cat_features(self) -> int:
        return len(self.cat_features_name)
    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y['train'])) if self.is_classification else 1
    @property
    def task_type(self) -> str: # binclass or regression or multiclass
        if self.is_classification:
            return 'binclass' if self.n_classes==2 else 'multiclass'
        else:
            return 'regression'
    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    @property
    def task_intro(self) -> str:
        return self.meta_data['task_intro']

    def size(self, part: str) -> int:
        X = self.N if self.N is not None else self.C
        assert X is not None
        return len(X[part])

    def text_serialization(self, feature, value) -> str:
        return f'{feature} is {value}' if value != np.nan else f'{feature} is [MASK]'

    def serialization_test(self, feature) -> str:
        return f'{feature} is'

    def get_length(self, key='train'):
        X = self.N if self.N is not None else self.C
        assert X is not None
        return len(X[key])

    def describe_full(self, key='train'):
        return [self.describe_item_full_by_index(i, key=key) for i in range(self.get_length(key=key))]

    def get_item(self, index, key='train'):
        if self.N is None:
            return (None, self.C[key][index], self.y[key][index])
        elif self.C is None:
            return (self.N[key][index], None, self.y[key][index])
        else:
            return (self.N[key][index], self.C[key][index], self.y[key][index])

    def get_item_unified(self, index, key='train'):
        return np.concatenate([x.astype(str).reshape(1, -1) for x in self.get_item(index, key) if x is not None], axis=1)

    def get_value(self, index, feature_index, key='train'):
        if feature_index == self.n_features:
            return self.y[key][index]
        elif feature_index < self.n_num_features:
            return self.N[key][index, feature_index]
        else:
            return self.C[key][index, feature_index-self.n_num_features]

class TPBERTaArgs:
    def __init__(self):
        self.result_dir = 'finetune_outputs'
        self.model_suffix = 'pytorch_models/best'
        self.dataset = 'HR Employee Attrition'
        self.lr = 1e-5
        self.weight_decay = 0.0
        self.max_epochs = 200
        self.early_stop = 50
        self.batch_size = 128
        self.task = 'binclass' # choices=['binclass', 'regression', 'multiclass']
        self.lamb = 0.0
        self.pretrain_dir = ''

class TPBERTa(object):
    def __init__(self, data_path, dataset_list, pretrain_dir, cache_path, device=torch.device('cuda')):
        self.datasets = {}
        self.dataset_configs = {}
        self.models = {}
        for name in tqdm(dataset_list):
            args = TPBERTaArgs()
            self.datasets[name] = TabularDataset.from_dir(os.path.join(data_path, name))
            self.datasets[name].name = name
            if self.datasets[name].is_classification:
                cur_pretrain_dir = os.path.join(pretrain_dir, 'tp-bin')
                args.task = 'multiclass' if self.datasets[name].n_classes > 2 else 'binclass'
            else:
                cur_pretrain_dir = os.path.join(pretrain_dir, 'tp-reg')
                args.task = 'regression'

        self.dataset_configs = DataConfig.from_pretrained(
            cur_pretrain_dir, 
            batch_size=args.batch_size, 
            preproc_type='lm',
            pre_train=False
        )

        args.pretrain_dir = str(cur_pretrain_dir) # pre-trained dir
        _, self.models = build_default_model(args, self.dataset_configs, 2, device, pretrain=True) # use pre-trained weights & configs
        self.models.eval()
        self.device = device
        self.cache_path = cache_path

    def encode_tables(self, tables):
        table_features = []
        for i_table in tables:
            extracted_table = {}
            name = i_table['from']
            table = i_table['value']
            header, data_rows = parse_table(table)

            num_split = 0
            extracted_mapping = {}
            if len(self.datasets[name].num_features_name) > 0:
                extracted_table['num'], extracted_mapping = extract_columns(header, data_rows, self.datasets[name].num_features_name)
                if self.datasets[name].N is not None:
                    # print(f'{torch.cuda.current_device()} extra {i_table}', extracted_table['num'])
                    self.datasets[name].N['add'] = extracted_table['num'].astype(float)
                    num_split = self.datasets[name].N['add'].shape[-1]

            if len(self.datasets[name].cat_features_name) > 0:
                extracted_table['cat'], extracted_mapping_cat = extract_columns(header, data_rows, self.datasets[name].cat_features_name)
                extracted_mapping.update({
                    k + num_split: v for k, v in extracted_mapping_cat.items()
                })
                if self.datasets[name].C is not None and extracted_table['cat'].shape[-1] > 0:
                    self.datasets[name].C['add'] = extracted_table['cat']

            data_loader, _ = prepare_tpberta_loaders(
                [self.datasets[name]],
                self.dataset_configs,
                tt=self.datasets[name].task_type,
                cache_path=self.cache_path,
                asked_keys=['add']
            )

            batch = next(iter(data_loader[0][0]['add']))

            if hasattr(self.models, 'module'):
                batch = {k: v.to(self.models.module.device) for k, v in batch.items()}
                IFA_emb, berta_emb = self.models.module.get_embedding(**batch)
            else:
                batch = {k: v.to(self.models.device) for k, v in batch.items()}
                IFA_emb, berta_emb = self.models.get_embedding(**batch)
            table_emb = reorder_tensor(IFA_emb[:, 1:, :], extracted_mapping) # TODO: or berta_emb
            table_features.append(table_emb)

        return table_features
