#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from wings.model.base_architecture import TabularMetaModel, WingsMetaForCausalLM

class TabularQwen2Config(Qwen2Config):
    model_type = "tabular_qwen2"

class TabularQwen2Model(TabularMetaModel, Qwen2Model):
    config_class = TabularQwen2Config

    def __init__(self, config: Qwen2Config):
        super(TabularQwen2Model, self).__init__(config)

class TabularQwen2ForCausalLM(Qwen2ForCausalLM, WingsMetaForCausalLM):
    config_class = TabularQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = TabularQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        tables: Optional[List] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_tabular_inputs(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                tables
            )

        cur = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        torch.cuda.empty_cache()
        return cur

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            attention_mask = attention_mask[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "tables": kwargs.get("tables", None),
            }
        )
        return model_inputs

    @classmethod
    def build(cls, model_name, model_path, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k in cls.MODEL_BUILD_KEYS}
        model = cls.from_pretrained(
            model_path,
            model_type=TabularQwen2Config.model_type,
            **model_kwargs
        )

        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k in cls.TOKENIZER_BUILD_KEYS}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            eos_token="<|im_end|>",
            **tokenizer_kwargs
        )
        return model, tokenizer


AutoConfig.register("tabular_qwen2", TabularQwen2Config)
AutoModelForCausalLM.register(TabularQwen2Config, TabularQwen2ForCausalLM)
