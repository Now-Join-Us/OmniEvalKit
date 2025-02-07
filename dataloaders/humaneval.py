from dataloaders.base import Dataset

class HumanEvalDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def get_reference(self, gold):
        test_func = gold["test"]
        entry_point = f"check({gold['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def preprocess_calculate_kwargs(self, base_kwargs, **kwargs):
        code_kwargs = base_kwargs.copy()
        code_kwargs['test_case'] = self.get_reference(base_kwargs["gold"])
        return code_kwargs

data_core = HumanEvalDataset