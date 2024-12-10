import os
import torch
import warnings
import sys

from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List

try:
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        expand2square,
        load_pretrained_model,
        tokenizer_image_token,
    )
    from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
except ImportError:
    warnings.warn(
        'Please install yi_vl from https://github.com/01-ai/Yi/tree/main/VL\nIf necessary, move the `llava` folder of the yi_vl repository to the executed folder')
    sys.exit(-1)

from evals.utils import place_begin_image_token
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper


class YiVL(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        key_info["model_path"] = model_path
        super().__init__()
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path)
        self.kwargs = {
            'temperature': 0.2,
            'top_p': None,
            'num_beams': 1,
        }
        self.model = self.model.to(dtype=torch.bfloat16)

    def preprocess_vqa(self, data):
        qs = place_begin_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens=[DEFAULT_IMAGE_TOKEN],
            leaved_token_num=1
        )
        conv = conv_templates['mm_default'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        conversation = {
            'conv': conv,
            'image_preloaded': data['image_preloaded'][0]
        }

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        prompt = conversation['conv'].get_prompt()
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        image = conversation['image_preloaded']
        if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = conversation['conv'].sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).to(self.model.device),
                do_sample=True,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
                **self.kwargs
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        response = outputs.strip()
        return response

    def get_llm(self):
        return self.model.model

    def generate_text_only_from_token_id(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        llm = self.get_llm()
        outputs = llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

model_core = YiVL
