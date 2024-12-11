import torch
from torch.utils.data import Dataset

import os
import io
import copy
import json
import random

from typing import Dict
import transformers
from PIL import Image

from wings.configs import DEFAULT_IMAGE_TOKEN, BEGIN_LINE, END_LINE, DATASET_NAME2PATH
from wings.utils import import_class_from_string, logger
from wings.arguments import DataArguments
from wings.model.conversation_formatter import ConversationFormatter
from wings.dataloader.collators import DataCollatorForSupervisedDataset


class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        conversation_formatter: ConversationFormatter
    ):
        super(SupervisedDataset, self).__init__()
        self.conversation_formatter = conversation_formatter

        data_list = []
        for dataset_name in data_args.data_name.split('|'):
            dataset = json.load(open(DATASET_NAME2PATH[dataset_name], "r"))
            if data_args.data_processor:
                dataset = import_class_from_string(data_args.data_processor).process_train(
                    dataset,
                    mode=data_args.data_mode,
                    cache_dir=data_args.processed_data_dir,
                    data_name=dataset_name
                )
            data_list.extend(dataset)
        random_seed_42 = random.Random(42)
        for i in range(10):
            random_seed_42.shuffle(data_list)

        self.tokenizer = tokenizer
        self.data_list = data_list
        self.data_args = data_args
        assert not data_args.only_image_data or not (data_args.image_text_data_ratio > 0)
        if data_args.only_image_data:
            if torch.cuda.current_device() == 0:
                logger.info(f'[only_image_data=True] Dataset length before: {len(self.data_list)}')
            self.data_list = [item for item in self.data_list if 'image' in item]
            if torch.cuda.current_device() == 0:
                logger.info(f'[only_image_data=True] Dataset length after: {len(self.data_list)}')
        elif data_args.image_text_data_ratio > 0:
            if torch.cuda.current_device() == 0:
                logger.info(f'[image_text_data_ratio={data_args.image_text_data_ratio}] Dataset length before: {len(self.data_list)}')

            multimodal_indices = [idx for idx, v in enumerate(self.data_list) if 'image' in v.keys()]
            text_only_indices = [idx for idx, v in enumerate(self.data_list) if 'image' not in v.keys()]
            multimodal_sampled_num = int(1000000 * data_args.image_text_data_ratio / (data_args.image_text_data_ratio + 1))
            text_only_sampled_num = int(1000000 * 1 / (data_args.image_text_data_ratio + 1))

            if torch.cuda.current_device() == 0:
                logger.info(f'[image_text_data_ratio={data_args.image_text_data_ratio}] Dataset length (multimodal): {len(multimodal_indices)}')
                logger.info(f'[image_text_data_ratio={data_args.image_text_data_ratio}] Dataset length (text-only): {len(text_only_indices)}')
                logger.info(f'[image_text_data_ratio={data_args.image_text_data_ratio}] Dataset sampled multimodal / text-only: {multimodal_sampled_num} / {text_only_sampled_num}')

            multimodal_sampled_indices = random.sample(multimodal_indices, multimodal_sampled_num)
            text_only_sampled_indices = random.sample(text_only_indices, text_only_sampled_num)
            self.data_list = [self.data_list[idx] for idx in text_only_sampled_indices + multimodal_sampled_indices]
            rng = random.Random(42)
            rng.shuffle(self.data_list)
            if torch.cuda.current_device() == 0:
                logger.info(f'[image_text_data_ratio={data_args.image_text_data_ratio}] Dataset length after: {len(self.data_list)}')
        else:
            if torch.cuda.current_device() == 0:
                logger.info(f'Dataset length before: {len(self.data_list)}')
            i_del = 0
            while i_del < len(self.data_list):
                if 'conversations' not in self.data_list[i_del]:
                    del self.data_list[i_del]
                else:
                    i_del += 1
            if torch.cuda.current_device() == 0:
                logger.info(f'Dataset length after: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data_list:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data_list:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.data_list[i]
        conversations = copy.deepcopy(sample["conversations"])

        has_image = 'image' in sample
        if has_image:
            image_path = sample['image']
            if isinstance(image_path, list):
                image_path = image_path[0]
            processor = self.data_args.image_processor
            default_image = Image.new('RGB', (processor.crop_size['height'], processor.crop_size['width']),
                                      color=tuple(int(x * 255) for x in processor.image_mean))  # a mean image

            # read image
            if not os.path.isfile(image_path):
                logger.warning(f'Image file does not exist: {image_path}')
                image = default_image
            else:
                image = Image.open(image_path).convert('RGB')

            # image process
            try:
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            except Exception as e:
                logger.warning(BEGIN_LINE)
                logger.warning(f'process image of {image_path} fail')
                logger.warning(f'exception is: {e}')
                logger.warning(END_LINE)
                image = default_image # a mean image
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]  # process

            # align image token position
            first_set_idx = -1
            for cur_idx, conversation in enumerate(conversations):
                if DEFAULT_IMAGE_TOKEN in conversation['value'] and conversation['from'] == 'human':
                    first_set_idx = cur_idx
                    break
                elif DEFAULT_IMAGE_TOKEN not in conversation['value'] and conversation['from'] == 'human':
                    first_set_idx = cur_idx
            if first_set_idx != -1:
                for cur_idx in range(len(conversations)):
                    if cur_idx != first_set_idx:
                        conversations[cur_idx]['value'] = conversations[cur_idx]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    else:
                        if DEFAULT_IMAGE_TOKEN in conversations[cur_idx]['value']:
                            first_occurrence = conversations[cur_idx]['value'].find(DEFAULT_IMAGE_TOKEN)
                            conversations[cur_idx]['value'] = conversations[cur_idx]['value'][:first_occurrence+7] + conversations[cur_idx]['value'][first_occurrence+7:].replace(DEFAULT_IMAGE_TOKEN, '')
                            conversations[cur_idx]['value'] = conversations[cur_idx]['value'].strip()
                        else:
                            conversations[cur_idx]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + conversations[cur_idx]['value']
            else:
                logger.warning('Error: data.py cannot find any conversation.')

        # Now, sources are singleton list, with the element being list of dicts with two keys: `from` and `value`
        if not has_image and self.data_args.is_multimodal:
            if self.data_args.only_image_data:
                assert 0, 'only_image_data error'
            crop_size = self.data_args.image_processor.crop_size
            image = torch.zeros(3, crop_size['height'], crop_size['width'])
            for conversation in conversations:
                conversation['value'] = conversation['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt, input_ids, labels = self.conversation_formatter.format(conversations)

        return dict(
            input_ids=input_ids,
            labels=labels,
            image=image
        )

def make_supervised_data_module(data_args: DataArguments,
                                tokenizer: transformers.PreTrainedTokenizer,
                                conversation_formatter: ConversationFormatter) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_args=data_args,
                                      tokenizer=tokenizer,
                                      conversation_formatter=conversation_formatter)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
