from dataloaders.base import Dataset
import numpy as np

class BBHDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def estimate(self, scores, categories, sub_categories):
        statistics = self.estimator.sum_or_avg(scores, categories=categories, sub_categories=sub_categories,
                                               e_type='avg')
        statistics['full']['acc'] = np.mean([statistics[i]['acc'] for i in statistics.keys() if i != 'full'])

        return statistics

data_core = BBHDataset