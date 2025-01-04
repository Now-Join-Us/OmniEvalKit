from dataloaders.general import Dataset

class HumanEvalDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    # def calculate(self, data, filtered_response, is_filtered, question_type, request_type, calculate_type):
    #     test_func = data["test"]
    #     entry_point = f"check({data['entry_point']})"
    #     refrences = "\n" + test_func + "\n" + entry_point #测试用例
    def get_reference(self, gold):
        test_func = gold["test"]
        entry_point = f"check({gold['entry_point']})"
        return "\n" + test_func + "\n" + entry_point
    def base2code_kwargs(self, base_kwargs):
        code_kwargs = base_kwargs.copy()
        code_kwargs['test_case'] = self.get_reference(base_kwargs["gold"])
        return code_kwargs

data_core = HumanEvalDataset