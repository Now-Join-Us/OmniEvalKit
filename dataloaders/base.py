import os
import string
from utils import load_json, load_image, detect_language
from prompts.base import translate_prompt, TYPE2LANGUAGE2PROMPT
import importlib
from abc import ABC, abstractmethod
from evals.metrics import EXACT_MATCH

from dataloaders.calculators import loglikelihood, multiple_choice, exact_match, code_eval
from dataloaders.estimators import sum_or_avg


class Dataset(object):
    """

    the class of dataset.

    dataset_file_path(str): dataset file (json) path

    """
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        self.data = load_json(dataset_file_path)
        self.dataset_name = dataset_name
        self.image_url = image_url
        self.preloaded_image_num = preloaded_image_num
        if rank is not None and world_size is not None:
            self.data = [self.data[i] for i in list(range(rank, len(self.data), world_size))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        if 'image_path' in sample.keys():
            sample['image_url'] = self.image_url
            if self.preloaded_image_num > 0:
                sample['image_preloaded'] = [load_image(os.path.join(sample['image_url'], sample['image_path'][idx])) for idx in range(self.preloaded_image_num)]

        if sample['prompt_instruction'] is not None:
            return sample

        language = detect_language(sample['raw_instruction'])
        if sample['question_type'] == 'multiple_choice':
            choices_str = 'Options:\n' + '\n'.join([
                f'{choice_id}. {to_prompt_choice}'
                for choice_id, to_prompt_choice in zip(
                    string.ascii_uppercase[:len(sample['choices'])],
                    sample['choices']
                )
            ])
            sample['prompt_instruction'] = f"{translate_prompt('Question: ', language)}{sample['raw_instruction']}\n" + \
                f"{choices_str}\n" + TYPE2LANGUAGE2PROMPT['multiple_choice'][language]
        elif sample['question_type'] == 'open':
            sample['prompt_instruction'] = f"{translate_prompt('Question: ', language)}{sample['raw_instruction']}\n" + TYPE2LANGUAGE2PROMPT['open'][language]
        elif sample['question_type'] == 'yes_or_no':
            sample['prompt_instruction'] = f"{translate_prompt('Question: ', language)}{sample['raw_instruction']}\n" + TYPE2LANGUAGE2PROMPT['yes_or_no'][language]
        else:
            raise NotImplementedError(f"Some pre-process may be required for {sample['question_type']}.")

        if 'hint' in sample.keys() and sample['hint'] is not None:
            sample['prompt_instruction'] = f"{translate_prompt('Hint: ', language)}{sample['hint']}\n" + sample['prompt_instruction']
        return sample

    @abstractmethod
    def preprocess_calculate_kwargs(self, base_kwargs, **kwargs):
        raise NotImplementedError

    def is_overridden_preprocess_calculate_kwargs(self, obj):
        return Dataset.__dict__['preprocess_calculate_kwargs'] is not obj.preprocess_calculate_kwargs.__func__

    def caculate(self, data, filtered_response, is_filtered, question_type, request_type, calculate_func, **inner_kwargs):
        base_calculate_kwargs = {
            'filtered_r': filtered_response,
            'is_filtered': is_filtered,
            'gold': data['gold']
        }
        if self.is_overridden_preprocess_calculate_kwargs(self):
            base_calculate_kwargs = self.preprocess_calculate_kwargs(base_calculate_kwargs)

        ## 如果用户有传就用传的 ##
        ## --eval_args calculate_func=xxx
        if calculate_func is not None:
            calculate_core = getattr(
                importlib.import_module(f'dataloaders.calculators'),
                calculate_func
            )
            return calculate_core(**base_calculate_kwargs, **inner_kwargs)

        ## 如果用户有定义 ##
        if question_type is None:
            question_type = data.get('question_type', 'open')
        if request_type is None:
            request_type = data.get('request_type', 'open')

        if question_type == 'multiple_choice':
            if request_type == 'loglikelihood':
                metric2score, filtered_base_dict = loglikelihood(
                    **base_calculate_kwargs,
                    choices_length=[len(i) for i in data.get('choices', data.get('prompt_choices', None))],
                    prompt_choices=data['prompt_choices'],
                    **inner_kwargs
                )
            else:
                metric2score = multiple_choice(**base_calculate_kwargs, **inner_kwargs)

        elif question_type == 'open':
            metric2score = exact_match(
                **{k: base_calculate_kwargs[k] for k in ['filtered_r', 'gold']},
                max_to_0_1=True,
                **EXACT_MATCH.get(data['name'], {}),
                **inner_kwargs
            )
        elif question_type == 'yes_or_no':
            metric2score = multiple_choice(**base_calculate_kwargs, **inner_kwargs)

        else:
            raise NotImplementedError(f'Unknown question_type: {question_type}')

        return metric2score

    def estimate(self, scores, categories, sub_categories, estimate_func, **inner_kwargs):
        if estimate_func is not None:
            estimate_core = getattr(
                importlib.import_module(f'dataloaders.estimators'),
                estimate_func
            )
            return estimate_core(scores, categories=categories, sub_categories=sub_categories, **inner_kwargs)
        return sum_or_avg(scores, categories=categories, sub_categories=sub_categories, e_type='avg', **inner_kwargs)

data_core = Dataset