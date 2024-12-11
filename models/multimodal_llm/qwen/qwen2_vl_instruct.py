from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base import ModelWrapper
import os

class Qwen2VL(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, **model_args
        )
        self.processor = AutoProcessor.from_pretrained(model_path, **tokenizer_args)

    def preprocess_vqa(self, data):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(data['image_url'], data['image_path'][0]),
                    },
                    {"type": "text", "text": data['prompt_instruction']},
                ],
            }
        ]

        return messages

    def generate_vqa(self, conversation):
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def generate_ppl(self, conversation):
        pass

    def get_llm(self):
        return self.model

model_core = Qwen2VL