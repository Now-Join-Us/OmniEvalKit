# This file includes functions adapted from the lm-evaluation-harness repository (https://github.com/EleutherAI/lm-evaluation-harness).
# Original work by Gao et al., licensed under MIT license.
# Copyright (c) 2020 EleutherAI

import importlib
from tqdm import tqdm

from utils import save_pickle, save_json, logger


class EvalTool(object):
    """Calculate metric

    Args:
        dataset_name(str): name of dataset
        dataset: object of dataset
        filter_type(dict): ways of extracting answers
        filter_model_wrapper: if the answer is extracted using the model, then it exists, indicating that the extracting model
    """
    def __init__(self,
        dataset_name, dataset,
        filter_type=None, filter_args={}, filter_model_wrapper=None,
        question_type=None, request_type=None, calculate_type=None, estimate_type='each_then_overall',
        **kwargs
        ):
        ## 构建 filter 模块 ##
        self.filters = []
        self.filter_types = filter_type.split(',')
        for f_type in self.filter_types:
            filter_module = importlib.import_module(f'evals.filters.{f_type}') # 等价于 from eval.infer.regex
            self.filters.append(getattr(filter_module, 'filter_core')(dataset_name=dataset_name, **filter_args)) # 等价于 from infer.regex import filter_core as center

        self.dataset_name = dataset_name
        self.dataset = dataset
        self.fallback = '[invalid]'
        self.question_type = question_type
        self.request_type = request_type
        self.calculate_type = calculate_type
        self.estimate_type = estimate_type

    def calculate_score(self, filtered_responses):
        """Calculate the correlation metric (score) for each

        Args:
            filtered_responses: filtered answer

        Returns:
            Contains two, one is the metric (score) for each; the other is answers (to save)
        """
        if self.estimate_type == 'overall':
            ## TODO: 对一整个数据集直接算指标的 ##
            scores = self.dataset.overall_calculate(self.dataset, filtered_responses)

        elif self.estimate_type == 'each_then_overall':
            scores = []
            for idx, (data, filtered_dict) in enumerate(tqdm(zip(self.dataset, filtered_responses), total=len(self.dataset))):
                metric2score = self.dataset.calculate(
                    data,
                    filtered_dict['filtered_response'],
                    filtered_dict['is_filtered'],
                    question_type=self.question_type,
                    request_type=self.request_type,
                    calculate_type=self.calculate_type
                )
                scores.append(metric2score)
            return scores
        else:
            raise NotImplementedError

    def estimate_statistic(self, scores):
        """Calculate metric for a dataset

        Args:
            scores(list): the relevant metrics for each

        Returns:
            metric for the entire dataset
        """
        if self.estimate_type == 'each_then_overall':
            categories = [data['category'] for data in self.dataset] if 'category' in self.dataset[0].keys() else None
            sub_categories = [data['sub_category'] for data in self.dataset] if 'sub_category' in self.dataset[0].keys() else None

            return self.dataset.estimate(scores, categories, sub_categories)
        else:
            raise NotImplementedError

    def filter_answer(self, responses):
        """Extract answer

        Args:
            dataset_name(str): answer of model

        Returns:
            Contains two, one is a boolean value, indicating whether the answer is extracted; the other is a string, indicating the extracted answer
        """

        filtered_responses = []
        for f in self.filters:
            for r, entry in zip(responses, self.dataset):
                filtered_responses.append(f.apply(response=r, data=entry, question_type=self.question_type))
        return filtered_responses

    def evaluate(self, responses, full_score_save_path, statistic_save_path):
        """integrate evaluation-related functions and the overall evaluation process

        Args:
            responses: answer of model
            full_score_save_path: the file storing the scores of each question
            statistic_save_path: the file storing the metrics of the dataset

        Returns:
            metric for the entire dataset
        """

        ## 0. 一些预处理 ##
        responses = [responses[data['id']] for data in self.dataset if data['id'] in responses.keys()] # 提取所有已经获得的 response
        if isinstance(responses[0], list): # 展平list
            responses = sum(responses, [])

        if any([data['gold'] is None for data in self.dataset]):
            logger.warning(f'警告：存在一些 gold 为 None，跳过 eval，直接保存结果：{full_score_save_path}')
            self.save(full_score_save_path, [{'response': i} for i in responses])
            return None

        ## 1. Filter 提取回复中的答案，例如从“这道题的答案是A，因为xxx。”中提取 'A' ##
        filtered_responses = self.filter_answer(responses)
        ## 2. Calculate 计算上一步提取的回复中的答案，与真实答案（Ground Truth）之间的 metric 值 ##
        scores = self.calculate_score(filtered_responses)
        ##    保存 calculate 后的 metric 值和所有数据 ##
        self.save(full_score_save_path, [{'response': i, **j, 'score': k} for i, j, k in zip(responses, filtered_responses, scores)], is_save_pickle=False)

        ## 3. 如果是**每条**数据都得到了一个测评的 metric 值，下面的函数将它们总体/每类上的指标计算出来（例如在类别子集上平均等） ##
        statistics = self.estimate_statistic(scores)
        ##    保存总体/每类的指标 ##
        save_json(statistic_save_path, statistics)
        return statistics

    def save(self, file_path, results, is_save_pickle=True):
        saved_results = [{**data, **cur_result} for data, cur_result in zip(self.dataset, results)]
        save_json(file_path, saved_results)
        if is_save_pickle:
            save_pickle(file_path[:file_path.rfind('.')] + '.pkl', saved_results)
