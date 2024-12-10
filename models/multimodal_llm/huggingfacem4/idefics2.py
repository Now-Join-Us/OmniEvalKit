import os
from transformers import AutoProcessor, AutoModelForVision2Seq
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from models.base import ModelWrapper


class Idefics2(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path, **tokenizer_args)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **model_args)

    def preprocess_vqa(self, data):
        instruction, images = data['prompt_instruction'], data['image_preloaded']

        messages, start_index = [], 0
        has_tag = False
        for tag in DATASET2DEFAULT_IMAGE_TOKEN[data['name']]:
            tag_pos = instruction.find(tag)

            if tag_pos != -1:
                has_tag = True
                if start_index < tag_pos:
                    messages.append({"type": "text", "text": instruction[start_index:tag_pos]})

                messages.append({"type": "image"})
                start_index = tag_pos + len(tag)
        if start_index < len(instruction):
            messages.append({"type": "text", "text": instruction[start_index:]})

        if not has_tag:
            messages = [{"type": "image"}] + messages

        conversation = (messages, images)
        return conversation

    def generate_vqa(self, conversation, **kwargs):
        messages, images = conversation
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return response

    def get_llm(self):
        return self.model

model_core = Idefics2
