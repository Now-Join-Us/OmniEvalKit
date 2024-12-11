#    Copyright (C) 2024 AIDC-AI
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers.utils import ModelOutput
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from dataclasses import dataclass

import math
import warnings
from typing import Optional, Tuple, Union, List

from wings.model.base_architecture import LlavaMetaModel, WingsMetaForCausalLM
from wings.model.base_module import ReweightLinear, WingsAttention


def _prepare_4d_causal_attention_mask_q_kv(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask

class WingsQwen2Config(Qwen2Config):
    model_type = "wings_qwen2"
    def __init__(self, **kwargs):
        super(WingsQwen2Config, self).__init__(**kwargs)

class WingsQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = WingsQwen2Config

    def __init__(self, config: Qwen2Config):
        super(WingsQwen2Model, self).__init__(config)

@dataclass
class ModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def wings_layer_forward(self):
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
        output_attentions: Optional[bool],
        use_cache: Optional[bool],
        image_features: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        position_ids_image: Optional[torch.Tensor],
        position_ids_text: Optional[torch.Tensor],
        padding_length: Optional[List] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache
        )

        hidden_states = residual + hidden_states
        hidden_states_wings = []
        if image_features is not None and not isinstance(image_features, list) and position_ids_image is not None and len(position_ids_image) != 0:
            hidden_states_image, _, _ = self.attn_pool(
                query=hidden_states,
                key=image_features,
                value=image_features,
                attention_mask=None,
                position_ids_q=position_ids,
                position_ids_image=position_ids_image,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states_wings.append(hidden_states_image)

        if hasattr(self, 'reweight_module') and text_features is not None:
            hidden_states_text, _, _ = self.attn_t_pool(
                query=hidden_states,
                key=text_features,
                value=text_features,
                attention_mask=attention_mask,
                position_ids_q=position_ids,
                position_ids_image=position_ids_text,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_length=padding_length
            )
            hidden_states_wings.append(hidden_states_text)

        if len(hidden_states_wings) > 1:
            reweight = self.reweight_module(self_attn_weights)
            if padding_length is not None:
                if all([i == 0 for i in padding_length]):
                    pad_mask2image = torch.zeros_like(reweight[0])
                else:
                    pad_mask2image = torch.ones_like(reweight[0])
                    pad_mask2image[[pad_mask_i for pad_mask_i, pad_mask_v in enumerate(padding_length) if pad_mask_v == 0], :, 0] = 0
                hidden_states = hidden_states + reweight[0] * hidden_states_wings[0] * pad_mask2image
            else:
                hidden_states = hidden_states + reweight[0] * hidden_states_wings[0]
            hidden_states = hidden_states + reweight[1] * hidden_states_wings[1]

            for reweight_i, hidden_state_i in zip(reweight, hidden_states_wings):
                hidden_states = hidden_states + reweight_i * hidden_state_i
        elif len(hidden_states_wings) == 1:
            hidden_states = hidden_states + hidden_states_wings[0]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    return forward

def wings_forward(self):
    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        position_ids_image: Optional[torch.Tensor] = None,
        position_ids_text: Optional[torch.Tensor] = None,
        use_cache_for_image: Optional[bool] = False,
        output_wings_loss: Optional[bool] = False,
        padding_length: Optional[List] = None
    ) -> Union[Tuple, ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = _prepare_4d_causal_attention_mask_q_kv(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_wings_loss = [] if output_wings_loss else None
        next_decoder_cache = None

        for cur_layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    image_features,
                    text_features,
                    position_ids_image,
                    position_ids_text,
                    padding_length
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    image_features=image_features,
                    text_features=text_features,
                    position_ids_image=position_ids_image,
                    position_ids_text=position_ids_text,
                    padding_length=padding_length
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_wings_loss:
                all_wings_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_wings_loss] if
                v is not None)

        return ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward

class WingsQwen2ForCausalLM(Qwen2ForCausalLM, WingsMetaForCausalLM):
    config_class = WingsQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = WingsQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def initialize_modules(self, model_args, data_args, training_args):
        self.config.lora_dim = model_args.lora_dim
        self.config.model_max_length = model_args.model_max_length
        self.config.system_prompt_length = model_args.system_prompt_length

        self.config.image_aspect_ratio = data_args.image_aspect_ratio
        self.config.image_token_length = data_args.image_token_length

        self.config.use_cache = training_args.use_cache
        self.config.tune_only_mm_mlp_adapter = training_args.tune_only_mm_mlp_adapter
        self.config.mm_projector_lr = training_args.mm_projector_lr
        self.config.vision_tower_lr_follow_mm_projector = training_args.vision_tower_lr_follow_mm_projector
        self.config.lr_projector_follow_tuned_keys = training_args.lr_projector_follow_tuned_keys

        for cur_layer_index in model_args.attn_layers_idx:
            self.model.layers[cur_layer_index].attn_pool = WingsAttention(self.config, cur_layer_index).to(torch.bfloat16)
            self.model.layers[cur_layer_index].attn_t_pool = WingsAttention(self.config, cur_layer_index).to(torch.bfloat16)
            if model_args.wings_router_type == 'linear':
                # TODO: find the hidden_size (4096)
                self.model.layers[cur_layer_index].reweight_module = ReweightLinear(4096).to(torch.bfloat16)

        for m in self.model.layers:
            m.forward = wings_layer_forward(m)

        self.model.forward = wings_forward(self.model)

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
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, i_t_indices, image_features
            ) = self.prepare_multimodal_inputs(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, get_image_features=True
            )
            text_features = inputs_embeds

        image_begin_end = []
        if i_t_indices is not None:
            for i_t_indices_pair in i_t_indices:
                base_index = 0
                for slice_idx, slice in enumerate(i_t_indices_pair):
                    if slice[1] != -1:
                        if slice[0] == 'i':
                            image_begin_end.append((base_index, base_index + self.config.image_token_length))
                            break
                        else:
                            base_index += slice[1]
                    else:
                        image_begin_end.append((0, self.config.image_token_length))

            assert len(i_t_indices) == len(image_begin_end)
        position_ids_image, position_ids_text = [], []
        padding_length = None
        if len(image_begin_end) > 0:
            position_ids_image.extend([
                torch.arange(
                    image_token_begin, image_token_end, dtype=torch.long, device=inputs_embeds.device
                ) for image_token_begin, image_token_end in image_begin_end
            ])
            position_ids_image = torch.stack(position_ids_image, dim=0)

            for image_token_begin, image_token_end in image_begin_end:
                temp_pos = torch.arange(0, inputs_embeds.shape[1])
                if image_token_begin == 0:
                    position_ids_text.append(temp_pos)
                else:
                    position_ids_text.append(torch.cat(
                        (temp_pos[self.config.system_prompt_length:image_token_begin], temp_pos[image_token_end:]), dim=0
                    ))
            max_length = max([len(i_l) for i_l in position_ids_text])
            is_padding = any([len(i_l) != max_length for i_l in position_ids_text])
            if is_padding:
                padding_length = [max_length - len(i) for i in position_ids_text]
                position_ids_text = [i if len(i) == max_length else F.pad(i, (0, pad_l), mode='constant', value=i[-1]) for pad_l, i in zip(padding_length, position_ids_text)]
            text_features = torch.stack([text_features[batch_idx, cur_t_indices, :] for batch_idx, cur_t_indices in enumerate(position_ids_text)])
            position_ids_text = torch.stack(position_ids_text, dim=0)
        else:
            position_ids_text = torch.arange(self.config.system_prompt_length, inputs_embeds.shape[1]).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
            text_features = torch.stack([text_features[batch_idx, cur_t_indices, :] for batch_idx, cur_t_indices in enumerate(position_ids_text)])

        if not isinstance(image_features, list) and inputs_embeds.shape[1] < image_features.shape[1]:
            image_features = image_features[:, :inputs_embeds.shape[1], :]
            padding_length = [0 for _ in range(image_features.shape[0])]
            if len(position_ids_image) != 0 and inputs_embeds.shape[1] < position_ids_image.shape[1]:
                position_ids_image = position_ids_image[:, :inputs_embeds.shape[1]]

        position_ids_text = position_ids_text.to(dtype=torch.long, device=inputs_embeds.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_features=image_features,
            text_features=text_features,
            position_ids_image=position_ids_image,
            position_ids_text=position_ids_text,
            padding_length=padding_length
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        assert loss is None or not torch.isnan(loss), "Error: Loss is NaN"

        torch.cuda.empty_cache()
        if not return_dict:
            output = (0,) + (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        assert past_key_values is None
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
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    @classmethod
    def build(cls, model_name, model_path, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k in cls.MODEL_BUILD_KEYS}
        model = cls.from_pretrained(
            model_path,
            model_type=WingsQwen2Config.model_type,
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

AutoConfig.register("wings_qwen2", WingsQwen2Config)
AutoModelForCausalLM.register(WingsQwen2Config, WingsQwen2ForCausalLM)
