from dataloaders.base import Dataset
import numpy as np

class MMEDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def estimate(self, scores, categories, sub_categories):
        raw_statistics = self.estimator.sum_or_avg(
            scores=scores,
            categories=None,
            sub_categories=sub_categories,
            e_type='avg'
        )
        shrunk_results = self.estimator.shrink_corresponding(scores, [data['shrink_pair'] for data in self],
                                                             categories, sub_categories)
        scores, categories, sub_categories = zip(
            *[({'acc': v['acc']}, v['category'], v['sub_category']) for v in shrunk_results.values()])
        scores, categories, sub_categories = list(scores), list(categories), list(sub_categories)
        shrunk_statistics = self.estimator.sum_or_avg(
            scores=scores,
            categories=None,
            sub_categories=sub_categories,
            e_type='avg'
        )

        sub_category2metric2score = {i: {'score': (raw_statistics[i]['acc'] + shrunk_statistics[i]['acc']) * 100} for i
                                     in raw_statistics.keys()}

        category_names = dict(
            perception=[
                'ocr', 'artwork', 'celebrity', 'color', 'count', 'existence', 'landmark', 'position', 'posters', 'scene'
            ],
            reasoning=[
                'code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation'
            ]
        )

        for cat in category_names.keys():
            sub_category2metric2score[cat] = {
                'score': sum([sub_category2metric2score[i]['score'] for i in category_names[cat]])}

        return sub_category2metric2score

data_core = MMEDataset