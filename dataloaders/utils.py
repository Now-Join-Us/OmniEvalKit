import re
from configs import DATASET2MODULE
import importlib

def detect_language(text):
    arabic_re = re.compile(r'[\u0600-\u06FF]')
    chinese_re = re.compile(r'[\u4e00-\u9fff]')
    english_re = re.compile(r'[a-zA-Z]')
    russian_re = re.compile(r'[\u0400-\u04FF]')

    if chinese_re.search(text):
        return 'ZH' # Mandarin Chinese (Simplified)
    elif arabic_re.search(text):
        return 'AR' # Arabic
    elif russian_re.search(text):
        return 'RU' # Russian

    return 'EN'

def translate_prompt(text, target_language):
    trans = {
        'Question: ': {
            'EN': 'Question: ',
            'ZH': '问题：',
            'AR': 'سؤال:',
            'RU': 'вопрос:'
        },
        'Hint: ': {
            'EN': 'Hint: ',
            'ZH': '提示：',
            'AR': 'تَلمِيح:',
            'RU': 'намекать:'
        },
        'Answer: ':{
            'EN': 'Answer: ',
            'ZH': '答案: ',
        },
        'Output: ':{
            'EN': 'Your output: ',
            'ZH': '你的输出: ',
        }
    }
    return trans[text][target_language]

def get_data(dataset_name, **kwargs):
    if dataset_name in DATASET2MODULE.keys():
        module = importlib.import_module(f'dataloaders.{DATASET2MODULE[dataset_name]}')
        data_core = getattr(module, 'data_core')
        dataset = data_core(dataset_name=dataset_name, **kwargs)
    else:
        from dataloaders.base import Dataset
        dataset = Dataset(dataset_name=dataset_name, **kwargs)

    return dataset
