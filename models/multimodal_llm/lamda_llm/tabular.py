from wings.utils import load_from_safetensors, set_seed
from wings.model.base_architecture import WingsMetaForCausalLM
from wings.arguments import ModelArguments, DataArguments, TrainingArguments

from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Tuple, Union

from models.base import ModelWrapper

import torch
import json

class Tabular(ModelWrapper):
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

        self.model.get_model().initialize_tabular_modules(
            model_args=local_model_args,
            data_args=data_args,
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

        self.model.get_model().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        self.model.lm_head.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    def preprocess_vqa(self, data):
        return (data['prompt_instruction'], [data['table']])

    def generate_vqa(self, conversation, **kwargs):
        instruction, tables = conversation
        prompt, input_ids = self.conversation_formatter.format_query(instruction)
        do_sample = False
        input_ids = input_ids.unsqueeze(0).cuda()
        with torch.inference_mode():
            infer_kwargs = dict(
                tables=tables,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                repetition_penalty=None,
                use_cache=True
            )
            output_ids = self.model.generate(
                input_ids,
                **infer_kwargs
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

    def preprocess_prompt_instruction(self, prompt_instruction, **kwargs):
        return prompt_instruction

    def get_generated(self, instruction, tables, **kwargs):
        prompt, input_ids = self.conversation_formatter.format_query(instruction)
        do_sample = False
        input_ids = input_ids.unsqueeze(0).cuda()
        infer_kwargs = dict(
            tables=tables,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            repetition_penalty=None,
            use_cache=True
        )
        return {'input_ids': input_ids}, infer_kwargs

model_core = Tabular
