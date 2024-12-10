from dataloaders.base import Dataset
import numpy as np

class MMCCDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def estimate(self, scores, categories, sub_categories):
        shrunk_results = self.estimator.shrink_corresponding(scores,
                                                             [data['shrink_pair'] for data in self],
                                                             categories, sub_categories)

        scores, categories, sub_categories = zip(
            *[({'acc': v['acc']}, v['category'], v.get('sub_category', None)) for v in shrunk_results.values()])

        scores, categories, sub_categories = list(scores), list(categories), list(sub_categories)

        shrunk_statistics = self.estimator.sum_or_avg(
            scores=scores,
            categories=categories,
            sub_categories=sub_categories,
            e_type='avg'
        )

        return shrunk_statistics

data_core = MMCCDataset