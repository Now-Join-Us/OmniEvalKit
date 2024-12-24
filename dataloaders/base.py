import os
import string
from utils import load_json, load_image
from dataloaders.utils import detect_language, translate_prompt
from configs import TYPE2LANGUAGE2PROMPT
from evals.calculators import BaseCalculator
from evals.estimators import BaseEstimator
from evals.metrics import EXTRACTMATCH
from evals.utils import opt_or_base_type

class Dataset(object):
    """

    the class of dataset.

    dataset_file_path(str): dataset file (json) path

    """
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        self.data = load_json(dataset_file_path)
        self.dataset_name = dataset_name
        self.calculator = BaseCalculator
        self.estimator = BaseEstimator
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

    def calculate(self, data, filtered_response, is_filtered, question_type, request_type, calculate_type):
        question_type = opt_or_base_type(question_type, data.get('question_type', 'open'))
        request_type = opt_or_base_type(request_type, data.get('request_type', 'open'))
        base_calculate_kwargs = {
            'filtered_r': filtered_response,
            'is_filtered': is_filtered,
            'gold': data['gold']
        }
        if question_type == 'multiple_choice':
            if request_type == 'loglikelihood':
                metric2score, filtered_base_dict = getattr(self.calculator, opt_or_base_type(calculate_type, 'loglikelihood'))(
                    **base_calculate_kwargs,
                    choices_length=[len(i) for i in data.get('choices', data.get('prompt_choices', None))],
                    prompt_choices=data['prompt_choices']
                )
            else:
                metric2score = getattr(self.calculator, opt_or_base_type(calculate_type, 'multiple_choice'))(**base_calculate_kwargs)

        elif question_type == 'open':
            gold = base_calculate_kwargs['gold']
            if isinstance(gold, int) and 'prompt_choices' in data.keys():
                gold = data['prompt_choices'][gold]
            metric2score = getattr(self.calculator, opt_or_base_type(calculate_type, 'exact_match'))(
                filtered_r=base_calculate_kwargs['filtered_r'],
                gold=gold,
                max_to_0_1=True,
                **EXTRACTMATCH.get(data['name'], {})
            )

        elif question_type == 'yes_or_no':
            metric2score = getattr(self.calculator, opt_or_base_type(calculate_type, 'multiple_choice'))(**base_calculate_kwargs)
        else:
            raise NotImplementedError(f'Unknown question_type: {question_type}')

        return metric2score

    def estimate(self, scores, categories, sub_categories):
        return self.estimator.sum_or_avg(scores, categories=categories, sub_categories=sub_categories, e_type='avg')

data_core = Dataset