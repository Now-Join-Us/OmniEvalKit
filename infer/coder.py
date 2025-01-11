import torch
from infer.base import InferCenter
import warnings
from typing import List
import os
INFILL_MODE = False
INSTRUCTION_MODE = False


class InferCodeCenter(InferCenter):
    """chain of thought inference

    Args:
        model_wrapper: object of model wrapper
    """
    
    def __init__(self, model_wrapper, **kwargs):
        # 要传进一个关于任务的参数
        super().__init__(model_wrapper)
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )
        self.dataset_name = kwargs.get("task_name", "humaneval")
        self.eos = [
            "<|endoftext|>",
            "<|endofmask|>",
            "</s>",
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
        ]
        self.instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
        self.response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(self.dataset_name)
        else:  # with chat template
            self.eos += ["\n```\n"]

        # print(f"{self.eos = }")
        # print(1)
        self.skip_special_tokens = True
        # if "humaneval" in kwargs.get("task_name", "humaneval"):
        #     self.stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]
        # else:
        #     pass
    def is_direct_completion(self) -> bool:
        return False
        # return self.force_base_prompt or self.model_wrapper.tokenizer.chat_template is None
    def codegen(
        self, data: str, device=torch.device('cuda'), **kwargs
    ) -> List[str]:
        
        prompt = data['prompt_instruction']
        # if self.temperature == 0:
        #     assert not do_sample
        #     assert num_samples == 1
        do_sample = kwargs.get('do_sample', False)
        num_samples = kwargs.get('num_samples', 1)
        max_new_tokens = kwargs.get('max_new_tokens', 768)
        if do_sample:
            top_p = kwargs.get('top_p', 0.95)
            
        # import pdb; pdb.set_trace()
        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.model_wrapper.tokenizer
            )
        )

        # import pdb; pdb.set_trace()
        input_tokens = self.model_wrapper.tokenizer.encode(prompt, return_tensors="pt").to(
            device
        )
        # kwargs = {}
        # if do_sample:
        #     kwargs["top_p"] = 0.95
        #     kwargs["temperature"] = self.temperature

        outputs = self.model_wrapper.model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences = num_samples,
            # num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.model_wrapper.tokenizer.pad_token_id or self.model_wrapper.tokenizer.eos_token_id,
            stop_strings=self.eos,
            tokenizer=self.model_wrapper.tokenizer,
            # **kwargs,
        )
        
        gen_strs = self.model_wrapper.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        # import pdb; pdb.set_trace()
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        # import pdb; pdb.set_trace()
        return outputs
    
    def infer(self, data, device=torch.device('cuda'), **kwargs):
        return self.codegen(data, device, **kwargs)

    def infer_coder(self, data, device=torch.device('cuda'), **kwargs):
        keys_to_extract = ["num_return_sequences", "do_sample", "temperature", "top_p", "top_k", "max_length"]
        gen_kwargs = {key: kwargs[key] for key in keys_to_extract if key in kwargs}
        # import pdb;pdb.set_trace()
        token_gens = self.model_wrapper.generate_k_tokens(
            conversation=data['prompt_instruction'],
            # num_return_sequences = RETURN_K ,
            **gen_kwargs,
        ) ## List len==RETURN_K
        # import pdb;pdb.set_trace()
        # INFILL_MODE = kwargs['INFILL_MODE']
        stop_words = self.stop_words
        # INSTRUCTION_MODE = kwargs['INSTRUCTION_MODE']
        postprocess = kwargs.get('postprocess', True)

        code_gens_result = []
        tokenizer = self.model_wrapper.tokenizer
        for generated_tokens in token_gens:
            # for s in generated_tokens:
            if INFILL_MODE or tokenizer.eos_token in stop_words:
                if generated_tokens[0] == tokenizer.bos_token_id:
                    generated_tokens = generated_tokens[1:]
                # Treat eos token as a regular stop word not removing it from the output
                # If it's removed it may have the effect of removing it in the middle of a
                # longer generation in case a batch size > 1 is used, which will result in
                # a wrong generation as it won't be used for splitting lateron
                gen_code = tokenizer.decode(
                    generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                try:
                    # some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
                    tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
                    if gen_code.startswith(f"{tokenizer_prefix}"):
                        gen_code = gen_code[len(tokenizer_prefix):].lstrip()
                except:
                    pass
                if INFILL_MODE:
                    gen_code = self._parse_infill(gen_code, tokenizer)
                if INSTRUCTION_MODE:
                    # gen_code = _parse_instruction(gen_code, instruction_tokens)
                    pass
            else:
                gen_code = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            if not INFILL_MODE:
                # gen_code = gen_code[len(prefix) :]
                pass
            if postprocess:
                code_gens_result.append(
                    self.postprocess_generation(gen_code, data['raw_prompt'])
                )
            else:
                # warnings.warn(
                #     "model output is not postprocessed, this might lower evaluation scores"
                # )
                code_gens_result.append(gen_code)
        return code_gens_result
    def _parse_infill(code, tokenizer):
        """Reorder infill code and remove remaining special tokens."""
        model_id = tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            prefix, suffix, infill = code.split("<|mask:0|>", 2)
            infill = infill.split("<|endofmask|>")[0]
        elif model_id in ["bigcode/santacoder"]:
            prefix, rest = code.split("<fim-suffix>", 1)
            suffix, infill = rest.split("<fim-middle>", 1)
            infill = infill.split("<|endoftext|>")[0]
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            prefix, rest = code.split("<fim_suffix>", 1)
            suffix, infill = rest.split("<fim_middle>", 1)
            infill = infill.split("<|endoftext|>")[0]
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")
        for k, v in tokenizer.special_tokens_map.items():
            if k == "additional_special_tokens":
                for t in v:
                    infill = infill.replace(t, "")
            else:
                infill = infill.replace(v, "")
        return infill

    def _parse_instruction(self, code, instruction_tokens):
        """Return code block after assistant_token/end_token"""
        _, end_token, assistant_token = instruction_tokens
        if not assistant_token and end_token:
            assistant_token = end_token
        elif not assistant_token and not end_token:
            return code

        idx = code.find(assistant_token)
        shift = len(assistant_token)
        if idx == -1:
            warnings.warn(
                "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
            )
            return code

        if "```python" in assistant_token:
            idx = code.find("```python", idx)
            shift = len("```python")
        return code[idx + shift :]

    def postprocess_generation(self, generation, raw_prompt):
            """Defines the postprocessing for a LM generation.
            :param generation: str
                code generation from LM
            :param idx: int
                index of doc in the dataset to which the generation belongs
                (not used for Humaneval-Task)
            """
            prompt = raw_prompt
            generation = generation[len(prompt) :]
            return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def _stop_at_stop_token(self, decoded_string, stop_tokens):
            """
            Produces the prefix of decoded_string that ends at the first occurrence of
            a stop_token.
            WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
            itself.
            """
            min_stop_index = len(decoded_string)
            for stop_token in stop_tokens:
                stop_index = decoded_string.find(stop_token)
                if stop_index != -1 and stop_index < min_stop_index:
                    min_stop_index = stop_index
            return decoded_string[:min_stop_index]
infer_core = InferCodeCenter

def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt

def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert"]
    raise ValueError(f"Unknown dataset: {dataset}")

# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
