# This file includes functions adapted from the tree-of-thought-llm repository (https://github.com/princeton-nlp/tree-of-thought-llm).
# Original work by Yao et al., licensed under MIT license.
# Copyright (c) 2023 Shunyu Yao

import itertools
import numpy as np
import torch
from infer.base import InferCenter
from copy import deepcopy


def value_prompt_wrap(data, y):
    data_copy = deepcopy(data)
    # TODO: 判断是拼 step 的 value_prompt 还是全局的 prompt
    value_prompt = "Please assess whether the response:\n{y}\nto the question:\n{prompt_instruction}\naligns with the ground truth:\n{gold}\n\nKindly offer one of \"sure\", \"likely\", or \"impossible\" on whether the response is correct. Judge: "
    data_copy['prompt_instruction'] = value_prompt.format(prompt_instruction=data['prompt_instruction'], y=y, gold=data['gold'])
    return data_copy

def value_outputs_unwrap(value_outputs: list) -> float:
    def check_string(s):
        return 0.001 if 'impossible' in s.lower() else (20 if 'sure' in s.lower() else 1)

    return sum(check_string(i) for i in value_outputs)

class InferToTCenter(InferCenter):
    """tree of thought inference

    Args:
        model_wrapper: object of model wrapper
    """
    def __init__(self, model_wrapper, steps=4,
                 generate_type='propose', evaluate_type='value', select_type='greedy',
                 n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5, **kwargs
                 ):
        super().__init__(model_wrapper)
        self.steps = steps
        self.generate_type = generate_type
        self.evaluate_type = evaluate_type
        self.select_type = select_type
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_select_sample = n_select_sample
        self.value_cache = {}

    def get_proposals(self, data, y):
        # TODO: 这里x包含y的话有个操作 拼的prompt好像不一样
        proposals = self.infer_chain_of_thought(data).split('\n')
        return [y + _ + '\n' for _ in proposals]

    def get_value(self, data, y, cache_value=True):
        value_prompt_data = value_prompt_wrap(data, y)
        value_prompt = value_prompt_data['prompt_instruction']
        if cache_value and value_prompt in self.value_cache:
            return self.value_cache[value_prompt]
        # value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
        value_outputs = [self.infer_direct(value_prompt_data) for _ in range(self.n_evaluate_sample)] # TODO: 要加 temperature 采样
        value = value_outputs_unwrap(value_outputs)
        if cache_value:
            self.value_cache[value_prompt] = value
        return value

    def get_values(self, data, ys, cache_value=True):
        values = []
        local_value_cache = {}
        for y in ys: # each partial output
            if y in local_value_cache: # avoid duplicate candidates
                value = 0
            else:
                value = self.get_value(data, y, cache_value=cache_value)
                local_value_cache[y] = value
            values.append(value)
        return values

    def infer(self, data, device=torch.device('cuda'), **kwargs):
        print('steps', self.steps)
        ys = [''] # current output candidates
        infos = []
        for step in range(self.steps):
            # generation
            if self.generate_type == 'propose':
                new_ys = [self.get_proposals(data, y) for y in ys]
            else:
                raise NotImplementedError
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))

            # evaluation
            if self.evaluate_type == 'value':
                values = self.get_values(data, new_ys)
            else:
                raise NotImplementedError

            # selection
            if self.select_type == 'sample':
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.n_select_sample, p=ps).tolist()
            elif self.select_type == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            infos.append({'step': step, 'instruction': data['prompt_instruction'], 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            ys = select_new_ys

        return ys, {'steps': infos}

infer_core = InferToTCenter
