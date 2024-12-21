# This file includes functions adapted from the lm-evaluation-harness repository (https://github.com/EleutherAI/lm-evaluation-harness).
# Original work by Gao et al., licensed under MIT license.
# Copyright (c) 2020 EleutherAI

from evals.filters import RegexFilter, ModelFilter

from utils import save_pickle, save_json

from tqdm import tqdm


class EvalTool(object):
    """Calculate metric

    Args:
        dataset_name(str): name of dataset
        dataset: object of dataset
        filter_type(dict): ways of extracting answers
        filter_model_wrapper: if the answer is extracted using the model, then it exists, indicating that the extracting model
    """
    def __init__(self, dataset_name, dataset, filter_type=None, filter_model_wrapper=None, **kwargs):
        self.filter_type = filter_type
        self.regex_filter = RegexFilter(dataset_name=dataset_name, **kwargs)
        if 'model' in filter_type:
            assert filter_model_wrapper is not None
            self.model_filter = ModelFilter(filter_model_wrapper, **kwargs)

        self.dataset_name = dataset_name
        self.dataset = dataset
        self.fallback = '[invalid]'

    def filter_answer(self, resps):
        """Extract answer

        Args:
            dataset_name(str): answer of model

        Returns:
            Contains two, one is a boolean value, indicating whether the answer is extracted; the other is a string, indicating the extracted answer
        """

        if self.filter_type == 'regex':
            return self.regex_filter.apply(resps, self.dataset)
        elif self.filter_type == 'model':
            return self.model_filter.apply(resps, self.dataset)
        elif self.filter_type == 'regex,model':
            regex_filtered_results = self.regex_filter.apply(resps, self.dataset)
            return self.model_filter.apply(resps, self.dataset, regex_filtered_results)
        elif self.filter_type == 'direct':
            return resps
        else:
            raise NotImplementedError

    def calculate_scores_each(self, resps, filtered_resps):
        """Calculate the correlation metric (score) for each

        Args:
            resps: answer of model
            filtered_resps: filtered answer

        Returns:
            Contains two, one is the metric (score) for each; the other is answers (to save)
        """
        scores, resps_to_save = [], []
        for idx, (data, r, filtered_dict) in enumerate(tqdm(zip(self.dataset, resps, filtered_resps), total=len(self.dataset))):
            filtered_r, is_filtered = filtered_dict['value'], filtered_dict['is_filtered']

            base_dict = {
                'filtered_r': filtered_r,
                'is_filtered': is_filtered
            }
            base_calculate_kwargs = {
                **base_dict,
                'gold': data['gold']
            }

            metric2score = self.dataset.caculate(data, base_dict, base_calculate_kwargs)

            scores.append(metric2score)
            resps_to_save.append({
                **base_dict,
                'response': r
            })
        return scores, resps_to_save

    def estimate_statistic(self, scores):
        """Calculate metric for a dataset

        Args:
            scores(list): the relevant metrics for each

        Returns:
            metric for the entire dataset
        """
        categories = [data['category'] for data in self.dataset] if 'category' in self.dataset[0].keys() else None
        sub_categories = [data['sub_category'] for data in self.dataset] if 'sub_category' in self.dataset[0].keys() else None

        return self.dataset.estimate(scores, categories, sub_categories)

    def evaluate(self, resps, full_score_save_path, statistic_save_path):
        """integrate evaluation-related functions and the overall evaluation process

        Args:
            resps: answer of model
            full_score_save_path: the file storing the scores of each question
            statistic_save_path: the file storing the metrics of the dataset

        Returns:
            metric for the entire dataset
        """
        resps = [resps[data['id']] for data in self.dataset if data['id'] in resps.keys()]
        if isinstance(resps[0], list):
            resps = sum(resps, [])

        if any([data['gold'] is None for data in self.dataset]):
            self.save(full_score_save_path, [{'score': i} for i in resps])
            return None

        filtered_answers = self.filter_answer(resps)
        scores, resps_to_save = self.calculate_scores_each(resps, filtered_answers)
        self.save(full_score_save_path, [{'score': i, **j} for i, j in zip(scores, resps_to_save)])

        statistics = self.estimate_statistic(scores)
        save_json(statistic_save_path, statistics)
        return statistics

    def save(self, file_path, results):
        saved_results = [{**data, **cur_result} for data, cur_result in zip(self.dataset, results)]
        save_json(file_path, saved_results)
        save_pickle(file_path[:file_path.rfind('.')] + '.pkl', saved_results)
