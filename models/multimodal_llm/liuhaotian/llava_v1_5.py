import re
import os
import torch
from models.base import ModelWrapper
import warnings
import sys


try:
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
    )
except ImportError:
    warnings.warn(
        'Please install llava from https://github.com/haotian-liu/LLaVA\nIf necessary, move the `LLaVA/llava` folder of the llava repository to the executed folder')
    sys.exit(-1)


class LLaVAv1_5(ModelWrapper):
    def __init__(self, model_path, model_args ,tokenizer_args, **kwargs):
        super().__init__()
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=self.model_name
        )

        self.generate_args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": self.model_name,
            "conv_mode": None,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

    def preprocess_vqa(self, data):
        qs = data['prompt_instruction']
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.generate_args.conv_mode is not None and conv_mode != self.generate_args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.generate_args.conv_mode, self.generate_args.conv_mode
                )
            )
        else:
            self.generate_args.conv_mode = conv_mode

        conv = conv_templates[self.generate_args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = data['image_preloaded']
        conversation = (prompt, images)
        return conversation

    def generate_vqa(self, conversation, **kwargs):
        prompt, images = conversation

        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.generate_args.temperature > 0 else False,
                temperature=self.generate_args.temperature,
                top_p=self.generate_args.top_p,
                num_beams=self.generate_args.num_beams,
                max_new_tokens=self.generate_args.max_new_tokens,
                use_cache=True,
            )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

model_core = LLaVAv1_5