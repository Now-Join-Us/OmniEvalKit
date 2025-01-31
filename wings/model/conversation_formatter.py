import torch

from typing import List, Dict
from abc import ABC, abstractmethod

from wings.configs import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_TABLE_TOKEN

class ConversationFormatter(ABC):

    @abstractmethod
    def format(self, conversations: List[Dict], generation_preface=None):
        pass

    @abstractmethod
    def format_query(self, query, generation_preface=""):
        pass


class ConversationFormatter(ConversationFormatter):
    def __init__(self, tokenizer, system_slot, user_slot, gpt_slot, eot, default_system_prompt="You are a helpful assistant."):
        self.tokenizer = tokenizer
        # self.from2role = {
        #     "system": "<|im_start|>system\n",
        #     "human": "<|im_start|>user\n",
        #     "gpt": "<|im_start|>assistant\n",
        # }
        self.from2role = {
            "system": system_slot,
            "human": user_slot,
            "gpt": gpt_slot,
        }
        self.eot = eot
        self.gpt_token_num = None
        self.default_system_prompt = default_system_prompt
        self.image_symbol = DEFAULT_IMAGE_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.ignore_index = IGNORE_INDEX

    def _tokenize_with_image_symbol(self, text):
        text_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in
                       text.split(self.image_symbol)]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(self.image_token_index)
        return token_ids

    def format(self, conversations: List[Dict], generation_preface=None):
        if self.gpt_token_num is None:
            self.gpt_token_num = len(self.tokenizer(self.from2role["gpt"], add_special_tokens=False).input_ids)

        if conversations[0]["from"] != "system":
            conversations.insert(0, {
                "from": "system",
                "value": self.default_system_prompt
            })

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = ""
        input_ids = []
        labels = []
        num_conversation = len(conversations)

        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"]
            text = role + message
            if i < num_conversation - 1 or generation_preface is None:
                text += self.eot
            if i < num_conversation - 1:
                text += '\n'
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_index] * len(token_ids)
            if frm == "gpt":
                label_ids[self.gpt_token_num:] = token_ids[self.gpt_token_num:]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface=""):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface)

        return prompt, input_ids

    def format_ppl_query(self, query, answer):
        prompt, input_ids, _ = self.format(
            [{
                "from": "human",
                "value": query
            },
            {
                "from": "gpt",
                "value": answer
            }], generation_preface=None)

        return prompt, input_ids


class TabularConversationFormatter(ConversationFormatter):
    def __init__(self, tokenizer, system_slot, user_slot, gpt_slot, eot, default_system_prompt="You are a helpful assistant."):
        super().__init__(tokenizer, system_slot, user_slot, gpt_slot, eot, default_system_prompt)
        self.image_symbol = DEFAULT_TABLE_TOKEN
