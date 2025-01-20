# Copyright (C) 2024 AIDC-AI
from typing import List, Tuple
import transformers
import torch


def tok_encode(
    tokenizer, string: str, left_truncate_len=None, add_special_tokens=None, add_bos_token=False
) -> List[int]:
    """ """
    # default for None - empty dict, use predefined tokenizer param
    # used for all models except for CausalLM or predefined value
    special_tokens_kwargs = {}

    # by default for CausalLM - false or self.add_bos_token is set
    if add_special_tokens is None:
        special_tokens_kwargs = {
            "add_special_tokens": False or add_bos_token
        }
    # otherwise the method explicitly defines the value
    else:
        special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

    encoding = tokenizer.encode(string, **special_tokens_kwargs)

    # left-truncate the encoded context to be at most `left_truncate_len` tokens long
    if left_truncate_len:
        encoding = encoding[-left_truncate_len:]

    return encoding

def tok_decode(tokenizer, tokens, skip_special_tokens=True):
    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

def tok_batch_encode(
    tokenizer,
    strings: List[str],
    padding_side: str = "left",
    left_truncate_len: int = None,
    truncation: bool = False,
    add_bos_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    add_special_tokens = {"add_special_tokens": False or add_bos_token}

    encoding = tokenizer(
        strings,
        truncation=truncation,
        padding="longest",
        return_tensors="pt",
        **add_special_tokens,
    )
    if left_truncate_len:
        encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
        encoding["attention_mask"] = encoding["attention_mask"][
            :, -left_truncate_len:
        ]
    tokenizer.padding_side = old_padding_side

    return encoding["input_ids"], encoding["attention_mask"]

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

def model_generate(model, tokenizer, context, max_length, stop, **generation_kwargs):
    # temperature = 0.0 if not set
    # if do_sample is false and temp==0.0:
    # remove temperature, as do_sample=False takes care of this
    # and we don't want a warning from HF
    generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
    do_sample = generation_kwargs.get("do_sample", None)

    # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
    if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
        generation_kwargs["do_sample"] = do_sample = False

    if do_sample is False and generation_kwargs.get("temperature") == 0.0:
        generation_kwargs.pop("temperature")
    # build stopping criteria
    stopping_criteria = stop_sequences_criteria(
        tokenizer, stop, context.shape[1], context.shape[0]
    )

    return model.generate(
        input_ids=context,
        max_length=max_length,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        **generation_kwargs,
    )

def encode_pair(tokenizer, context, continuation):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole_enc = tok_encode(tokenizer, context + continuation)
    context_enc = tok_encode(tokenizer, context)

    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]

    return context_enc, continuation_enc
