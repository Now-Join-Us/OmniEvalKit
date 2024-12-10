import torch
from transformers import AutoModelForCausalLM
from models.base import ModelWrapper
from configs import DATASET2DEFAULT_IMAGE_TOKEN
from evals.utils import replace_image_token


class Ovis1_6_Gemma2(ModelWrapper):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, **kwargs):
        super().__init__()
        assert model_path is not None
        self.model_path = model_path

        if 'device_map' in kwargs:
            if 'cuda' in kwargs['device_map']:
                self.device = torch.cuda.current_device()
            elif 'cpu' in kwargs['device_map']:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        self.dtype = kwargs['torch_dtype'] if 'torch_dtype' in kwargs else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=8192,
            trust_remote_code=True
        )

        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.max_partition = 9
        self.image_placeholder = '<image>'

        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )

    def preprocess_vqa(self, data):

        query = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=DATASET2DEFAULT_IMAGE_TOKEN[data['name']],
            target_tokens=self.image_placeholder,
            leaved_token_num=len(data['image_path'])
        )

        return query, data['image_preloaded']

    def generate_vqa(self, conversation, **kwargs):

        query, images = conversation

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=self.max_partition
        )

        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        pixel_values = [
            pixel_values.to(device=self.device, dtype=self.dtype) if pixel_values is not None else None
        ]

        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response

    def get_llm(self):
        return None

model_core = Ovis1_6_Gemma2
