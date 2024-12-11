# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from wings.utils import apply_rotary_pos_emb_with_position_ids, logger

import math
from typing import List, Optional, Tuple, Optional


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class ReweightLinear(nn.Module):
    def __init__(self, input_dim, act_fn: Optional[str] = 'silu', layer_idx: Optional[int] = None):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.act_fn = ACT2FN[act_fn]

    def forward(self, self_attn_weights, split_size_or_sections=1):
        bsz2token_attn = torch.sum(self_attn_weights, dim=1)

        bsz2token_attn = self.act_fn(F.linear(
            input=bsz2token_attn,
            weight=self.linear.weight[:, :bsz2token_attn.shape[-1]],
            bias=self.linear.bias)
            )

        bsz2token_attn = bsz2token_attn.softmax(dim=-1)
        splits = torch.split(bsz2token_attn, split_size_or_sections=split_size_or_sections, dim=-1)
        return list(splits)


class WingsAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        if self.num_key_value_heads * self.head_dim != self.hidden_size:
            self.num_key_value_heads = self.num_heads

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.config.lora_dim, bias=False),
            nn.Linear(self.config.lora_dim, self.num_heads * self.head_dim, bias=False)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.config.lora_dim, bias=False),
            nn.Linear(self.config.lora_dim, self.num_key_value_heads * self.head_dim, bias=False)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.config.lora_dim, bias=False),
            nn.Linear(self.config.lora_dim, self.num_key_value_heads * self.head_dim, bias=False)
        )
        self.o_proj = nn.Sequential(
            nn.Linear(self.num_heads * self.head_dim, self.config.lora_dim, bias=False),
            nn.Linear(self.config.lora_dim, self.hidden_size, bias=False)
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Sequential):
            trunc_normal_(m[0].weight, std=.02)
            nn.init.zeros_(m[1].weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids_q: Optional[torch.LongTensor] = None,
        position_ids_image: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_length: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            logger.warning(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = query.size()
        _, k_len, _ = key.size()

        query_states = query + self.q_proj(query)
        key_states = key + self.k_proj(key)
        value_states = value + self.v_proj(value)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q_seq_len = query_states.shape[-2]
        kv_seq_len = key_states.shape[-2]

        cos, sin = self.rotary_emb(query_states, seq_len=q_seq_len)
        query_states, key_states = apply_rotary_pos_emb_with_position_ids(
            q=query_states,
            k=key_states,
            cos=cos,
            sin=sin,
            position_ids_q=position_ids_q,
            position_ids_image=position_ids_image
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attention_mask = attention_mask[..., :kv_seq_len]
            if padding_length is not None and any([i != 0 for i in padding_length]):
                attention_mask = torch.stack([
                    torch.cat(
                        [attention_mask[i_pad, :, :-cur_pad, :],
                        torch.full((1, cur_pad, kv_seq_len), torch.finfo(attention_mask.dtype).min, device=attention_mask.device, dtype=attention_mask.dtype)],
                        dim=-2) if cur_pad != 0 else attention_mask[i_pad, :, :, :] for i_pad, cur_pad in enumerate(padding_length)
                ], dim=0)

            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

