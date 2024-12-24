# This file includes functions adapted from the VLMEvalKit repository (https://github.com/open-compass/VLMEvalKit).
# Original work by Duan et al., licensed under Apache-2.0 license.
# Copyright 2023 VLMEvalKit Authors. All rights reserved.
from dataloaders.base import Dataset

class OCRDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_file_path=None, rank=None, world_size=None, image_url=None, preloaded_image_num=1):
        super().__init__(dataset_name=dataset_name, dataset_file_path=dataset_file_path, rank=rank, world_size=world_size, image_url=image_url, preloaded_image_num=preloaded_image_num)

    def calculate(self, data, base_dict, base_calculate_kwargs):

        if data['category'] == 'handwritten_mathematical_expression_recognition':
            for j in range(len(base_calculate_kwargs['gold'])):
                answer = base_calculate_kwargs['gold'][j].strip().replace('\n', ' ').replace(' ', '')
                predict = base_calculate_kwargs['filtered_r'].strip().replace('\n', ' ').replace(' ', '')
                if answer in predict:
                    return {"acc": 1}
        else:
            for j in range(len(base_calculate_kwargs['gold'])):
                answer = base_calculate_kwargs['gold'][j].lower().strip().replace('\n', ' ')
                predict = base_calculate_kwargs['filtered_r'].lower().strip().replace('\n', ' ')
                if answer in predict:
                    return {"acc": 1}

        return {'acc': 0}


    def estimate(self, scores, categories, sub_categories):
        ocrbench_score = {}
        for i in range(len(scores)):
            if categories[i] not in ocrbench_score:
                ocrbench_score[categories[i]] = 0
            if scores[i]['acc'] > 0:
                ocrbench_score[categories[i]] += 1

        final_score_dict = {}
        if 'text_recognition' not in final_score_dict:
            final_score_dict['text_recognition'] = {}

        final_score_dict['text_recognition']['score'] = \
            (ocrbench_score['regular_text_recognition'] + ocrbench_score['irregular_text_recognition']
             + ocrbench_score['artistic_text_recognition'] + ocrbench_score['handwriting_recognition']
             + ocrbench_score['digit_string_recognition'] + ocrbench_score['non_semantic_text_recognition'])

        if 'scene_text_centric_vqa' not in final_score_dict:
            final_score_dict['scene_text_centric_vqa'] = {}

        final_score_dict['scene_text_centric_vqa']['score'] = ocrbench_score['scene_text_centric_vqa']

        if 'doc_oriented_vqa' not in final_score_dict:
            final_score_dict['doc_oriented_vqa'] = {}

        final_score_dict['doc_oriented_vqa']['score'] = ocrbench_score['doc_oriented_vqa']

        if 'key_information_extraction' not in final_score_dict:
            final_score_dict['key_information_extraction'] = {}

        final_score_dict['key_information_extraction']['score'] = ocrbench_score['key_information_extraction']

        if 'handwritten_mathematical_expression_recognition' not in final_score_dict:
            final_score_dict['handwritten_mathematical_expression_recognition'] = {}

        final_score_dict['handwritten_mathematical_expression_recognition']['score'] = \
            (ocrbench_score['handwritten_mathematical_expression_recognition'])

        if 'full' not in final_score_dict:
            final_score_dict['full'] = {}

        final_score_dict['full']['score'] = \
            (final_score_dict['text_recognition']['score'] + final_score_dict['scene_text_centric_vqa']['score']
             + final_score_dict['doc_oriented_vqa']['score'] + final_score_dict['key_information_extraction']['score']
             + final_score_dict['handwritten_mathematical_expression_recognition']['score'])

        final_score_dict['full']['score_norm'] = (float(final_score_dict['full']['score']) / 10)

        return final_score_dict

data_core = OCRDataset