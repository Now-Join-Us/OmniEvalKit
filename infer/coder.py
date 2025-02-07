import torch
from infer.base import InferCenter
import warnings
from typing import List
import os

class InferCodeCenter(InferCenter):
    """chain of thought inference

    Args:
        model_wrapper: object of model wrapper
    """
    
    def __init__(self, model_wrapper, **kwargs):
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


        self.skip_special_tokens = True

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
            
        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.model_wrapper.tokenizer
            )
        )


        input_tokens = self.model_wrapper.tokenizer.encode(prompt, return_tensors="pt").to(
            device
        )

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
        outputs = []

        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))

        return outputs
    
    def infer(self, data, device=torch.device('cuda'), **kwargs):
        return self.codegen(data, device, **kwargs)

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
