from wings.utils import load_from_safetensors, set_seed
from wings.model.base_architecture import WingsMetaForCausalLM
from wings.arguments import ModelArguments, DataArguments, TrainingArguments

from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Optional, Tuple, Union, List

from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper
from evals.utils import replace_image_token

import os
import torch
import json

from PIL import Image

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

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

class Wings(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__()
        set_seed(42)

        with open(model_args['json_path']) as json_file:
            config = json.load(json_file)

        local_model_args = ModelArguments(**config['model_args'])
        data_args = DataArguments(**config['data_args'])
        training_args = TrainingArguments(**config['training_args'])

        self.model, self.tokenizer, self.conversation_formatter = WingsMetaForCausalLM.build(
            model_name=local_model_args.model_name,
            model_path=local_model_args.model_path,
            conversation_formatter_kwargs={
                'system_slot': local_model_args.system_slot,
                'user_slot': local_model_args.user_slot,
                'gpt_slot': local_model_args.gpt_slot,
                'eot': local_model_args.eot
            },
            model_max_length=local_model_args.model_max_length
        )

        self.model.get_model().initialize_vision_modules(
            model_args=local_model_args,
            fsdp=training_args.fsdp
        )

        if hasattr(self.model, 'initialize_modules'):
            self.model.initialize_modules(
                model_args=local_model_args,
                data_args=data_args,
                training_args=training_args,
            )
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_max_length = self.tokenizer.model_max_length

        if local_model_args.model_safetensors_load_path is not None:
            self.model.load_state_dict(load_from_safetensors(self.model, local_model_args.model_safetensors_load_path))

        vision_tower = self.model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        self.model.to(torch.bfloat16)

    def preprocess_vqa(self, data):
        instruction = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens='<image>',
            leaved_token_num=1
        )

        conversation = (instruction, data['image_preloaded'][0])

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        instruction, image = conversation
        image_processor = getattr(self.model.get_vision_tower(), 'image_processor', None)
        if image is not None:
            image_tensor = process_images([image], image_processor, self.model.config).cuda()
        else:
            image_tensor = None

        prompt, input_ids = self.conversation_formatter.format_query(instruction)
        do_sample = False
        input_ids = input_ids.unsqueeze(0).cuda()
        with torch.inference_mode():
            kwargs = dict(
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=32,
                repetition_penalty=None,
                use_cache=True
            )
            output_ids = self.model.generate(
                input_ids,
                **kwargs
                )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        response = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        return response

    def generate_text_only_from_token_id(
        self,
        input_ids: torch.LongTensor = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        llm = self.get_llm()
        outputs = llm(input_ids=input_ids)

        return CausalLMOutputWithPast(
            logits=outputs[1]
        )

model_core = Wings
