from dataloaders.base import Dataset
import numpy as np
from collections import defaultdict

class HallusionDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def estimate(self, scores, categories, sub_categories):

        def hallusionbench(
                scores,
                categories,
                sub_categories,
                set_indices,
                figure_indices,
                question_indices,
                e_type='q_metric'
        ):
            category2metric2static = {}

            def _calculate(inputs, cal_categories, cal_set, cal_figure, cal_question):
                assert len(inputs) != 0
                metric2static = {}
                collected_metrics = sorted(list(set(inputs[0].keys())))
                for metric in collected_metrics:
                    res = defaultdict(list)

                    for i in range(len(inputs)):
                        if e_type == 'q_metric':
                            key = f"{cal_categories[i]}_{cal_set[i]}_{cal_question[i]}"
                            metric_key = 'q_' + metric
                        elif e_type == 'f_metric':
                            key = f"{cal_categories[i]}_{cal_set[i]}_{cal_figure[i]}"
                            metric_key = 'f_' + metric

                        res[key].append(inputs[i][metric])
                    metric2static[metric_key] = np.mean([np.all(x) for x in res.values()]) * 100

                return metric2static

            considered_categories = ['full'] + list(set(categories)) + list(set(sub_categories))
            considered_categories = sorted(considered_categories)

            for cat in considered_categories:
                if cat == 'full':
                    masked_scores = scores
                    masked_set_indices, masked_figure_indices, masked_question_indices = set_indices, figure_indices, question_indices
                    masked_categories = sub_categories
                else:
                    def _get_masked_first_item(l):
                        if cat in list(set(categories)):
                            return [i_first for i_first, m_cat in zip(l, categories) if cat == m_cat]
                        else:
                            return [i_first for i_first, m_cat in zip(l, sub_categories) if cat == m_cat]

                    masked_scores = _get_masked_first_item(scores)
                    masked_categories = _get_masked_first_item(sub_categories)
                    masked_set_indices, masked_figure_indices, masked_question_indices = \
                        _get_masked_first_item(set_indices), _get_masked_first_item(
                            figure_indices), _get_masked_first_item(question_indices)

                category2metric2static[cat] = _calculate(masked_scores, masked_categories, masked_set_indices,
                                                         masked_figure_indices, masked_question_indices)
            return category2metric2static

        statistics = self.estimator.sum_or_avg(scores=scores, categories=categories, sub_categories=sub_categories,
                                               e_type='avg')
        set_indices = [data['set_id'] for data in self]
        figure_indices = [data['figure_id'] for data in self]
        question_indices = [data['question_id'] for data in self]

        q_statistics = hallusionbench(
            scores=scores,
            categories=categories,
            set_indices=set_indices,
            figure_indices=figure_indices,
            question_indices=question_indices,
            sub_categories=sub_categories,
            e_type='q_metric'
        )

        f_statistics = hallusionbench(
            scores=scores,
            categories=categories,
            set_indices=set_indices,
            figure_indices=figure_indices,
            question_indices=question_indices,
            sub_categories=sub_categories,
            e_type='f_metric'
        )

        for key in f_statistics:
            statistics[key].update(q_statistics[key])
            statistics[key].update(f_statistics[key])

        for key in statistics:
            statistics[key]['acc'] *= 100

        return statistics

data_core = HallusionDataset