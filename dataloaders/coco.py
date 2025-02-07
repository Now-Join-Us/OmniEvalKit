from dataloaders.base import Dataset


class COCODataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def caculate(self, data, base_dict, base_calculate_kwargs):

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
            # (Meteor(), "METEOR"), # need java version 11.0.16+
            # (Spice(), "SPICE"), # need java version 11.0.16+
        ]

        total_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score({'0': [str(base_calculate_kwargs['gold'])]}, {'0': [str(base_calculate_kwargs['filtered_r'])]})
            if type(method) == list:
                for i in range(4):
                    total_scores[f'Bleu_{i + 1}'] = score[i] * 100
            else:
                total_scores[method] = score * 100

        return total_scores

data_core = COCODataset