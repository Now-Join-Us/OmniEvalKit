import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from models.base import tok_encode, tok_decode, tok_batch_encode, _model_generate
from dataloaders.utils import detect_language

from utils import get_max_length
from configs import MAX_GEN_TOKS, GEN_DATASET2UNTIL, GEN_DO_SAMPLE, GEN_TEMPERATURE, TYPE2LANGUAGE2PROMPT

def _encode_pair(tokenizer, context, continuation):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole_enc = tok_encode(tokenizer, context + continuation)
    context_enc = tok_encode(tokenizer, context)

    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]

    return context_enc, continuation_enc


def get_multiple_choice_encode(tokenizer, data, device=torch.device('cuda')):
    contexts, choices = data['prompt_instruction'], data['prompt_choices']
    if isinstance(contexts, str):
        contexts = [contexts] * len(choices)

    resp = []
    context_encode_list, continuation_encode_list = [], []
    for context, continuation in zip(contexts, choices):
        if context == "":
            # BOS or EOS as context
            context_enc, continuation_enc = (
                [tokenizer.eos_token_id],
                tok_encode(tokenizer, continuation),
            )
        else:
            context_enc, continuation_enc = _encode_pair(tokenizer, context, continuation)
        context_encode_list.append(context_enc)
        continuation_encode_list.append(continuation_enc)

    return context_encode_list, continuation_encode_list


class InferCenter(object):
    """methods for model inference

    Args:
        model_wrapper: object of model wrapper
    """
    def __init__(self, model_wrapper, **kwargs):
        self.model_wrapper = model_wrapper

    def generate_by_self(self, generate_type, data, device=torch.device('cuda')):
        type2func = {
            'text_only': self.model_wrapper.generate_text_only if hasattr(self.model_wrapper, 'generate_text_only') else None,
            'vqa': self.model_wrapper.generate_vqa if hasattr(self.model_wrapper, 'generate_vqa') else None
        }
        return type2func[generate_type](
            data['prompt_instruction'],
            device=device,
            aux_data=data
        )

    def generate_text_only(self, data, device=torch.device('cuda')):
        model, tokenizer = self.model_wrapper.get_llm(), self.model_wrapper.tokenizer
        context = data['prompt_instruction']
        max_length = get_max_length(model, tokenizer)

        # we assume all gen kwargs in the batch are the same
        # this is safe to assume because the `grouper` object ensures it.
        # unpack our keyword arguments.
        # add EOS token to stop sequences
        until = [tok_decode(tokenizer, tokenizer.eos_token_id, skip_special_tokens=False)]
        if data['name'] in GEN_DATASET2UNTIL.keys():
            until = GEN_DATASET2UNTIL[data['name']] + until
        max_gen_toks = MAX_GEN_TOKS
        max_ctx_len = max_length - max_gen_toks

        contexts = (context,) # batch: 1
        truncation = False
        # encode, pad, and truncate contexts for this batch
        context_enc, attn_masks = tok_batch_encode(
            tokenizer,
            contexts,
            left_truncate_len=max_ctx_len,
            truncation=truncation
        )
        context_enc = context_enc.to(device)
        attn_masks = attn_masks.to(device)

        # perform batched generation
        with torch.no_grad():
            model.eval()
            cont = _model_generate(
                model,
                tokenizer,
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                do_sample=GEN_DO_SAMPLE,
                temperature=GEN_TEMPERATURE,
                max_length=context_enc.shape[1] + max_gen_toks
            )

        resp = []
        cont_toks_list = cont.tolist()
        for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LM
            cont_toks = cont_toks[context_enc.shape[1] :]

            s = tok_decode(tokenizer, cont_toks)

            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            for term in until:
                if len(term) > 0:
                    # ignore '' separator,
                    # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                    s = s.split(term)[0]

            resp.append(s)
        return resp

    def loglikelihood_vqa(self, data, device=torch.device('cuda')):
        contexts, choices = data['prompt_instruction'], data['prompt_choices']
        if isinstance(contexts, str):
            contexts = [contexts] * len(choices)

        resp = []
        _, image = self.model_wrapper.preprocess_vqa(data)
        for context, continuation in zip(contexts, choices):
            whole_instruction = self.model_wrapper.preprocess_prompt_instruction(
                prompt_instruction=context + continuation,
                dataset_name=data['name'],
                img_num=len(data['image_path'])
            )
            context_instruction = self.model_wrapper.preprocess_prompt_instruction(
                prompt_instruction=context,
                dataset_name=data['name'],
                img_num=len(data['image_path'])
            )

            with torch.no_grad():
                self.model_wrapper.eval()
                whole_enc, _ = self.model_wrapper.get_generated(whole_instruction, image)
                context_enc, _ = self.model_wrapper.get_generated(context_instruction, image)

                outputs = self.model_wrapper.model(
                    **whole_enc,
                )
                logits = F.log_softmax(outputs.logits, dim=-1)

            logits = logits[:, context_enc['input_ids'].shape[-1] : , :] # [1, seq, vocab]

            continuation_enc = whole_enc['input_ids'][:, context_enc['input_ids'].shape[-1]:]
            logits = torch.gather(logits, 2, continuation_enc.unsqueeze(-1)).squeeze(-1)  # [1, seq]
            answer = (float(logits.sum()), False)
            resp.append(answer)

        return [resp]

    def loglikelihood_text_only(self, data, device=torch.device('cuda')):
        model, tokenizer = self.model_wrapper.get_llm(), self.model_wrapper.tokenizer
        contexts, choices = data['prompt_instruction'], data['prompt_choices']
        if isinstance(contexts, str):
            contexts = [contexts] * len(choices)
        max_length = get_max_length(model, tokenizer)

        resp = []
        for context, continuation in zip(contexts, choices):
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [tokenizer.eos_token_id],
                    tok_encode(tokenizer, continuation),
                )
            else:
                context_enc, continuation_enc = _encode_pair(tokenizer, context, continuation)

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(max_length + 1) :][:-1],
                dtype=torch.long,
                device=device,
            )
            (inplen,) = inp.shape

            with torch.no_grad():
                model.eval()
                if self.model_wrapper.is_overridden_generate_text_only_from_token_id(self.model_wrapper):
                    outputs = self.model_wrapper.generate_text_only_from_token_id(inp.unsqueeze(0))
                else:
                    outputs = model(inp.unsqueeze(0))

                logits = F.log_softmax(outputs.logits, dim=-1)

            logits = logits[:, len(inp) - len(continuation_enc) : len(inp), :] # [1, seq, vocab]

            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)

            continuation_enc = torch.tensor(
                continuation_enc, dtype=torch.long, device=device
            ).unsqueeze(0)
            max_equal = (greedy_tokens == continuation_enc).all()
            logits = torch.gather(logits, 2, continuation_enc.unsqueeze(-1)).squeeze(-1)  # [1, seq]
            answer = (float(logits.sum()), bool(max_equal))
            resp.append(answer)

        return [resp]

    def generate_vqa(self, data, device=torch.device('cuda')):
        if hasattr(self.model_wrapper, 'preprocess_vqa'):
            conversation = self.model_wrapper.preprocess_vqa(data) # MLLM on multimodal data
            return self.model_wrapper.generate_vqa(conversation)
        else:
            return self.generate_text_only(self.model_wrapper, data, device) # LLM on multimodal data

    def infer_direct(self, data, device=torch.device('cuda')):
        is_multimodal = 'image_path' in data.keys()

        if self.model_wrapper.force_use_generate or isinstance(data, list):
            return self.generate_by_self('vqa' if is_multimodal else 'text_only', data, device)

        if data['request_type'] == 'loglikelihood':
            if is_multimodal:
                return self.loglikelihood_vqa(data, device)
            return self.loglikelihood_text_only(data, device)
        elif data['request_type'] == 'generate_until':
            if is_multimodal:
                return self.generate_vqa(data, device) # MLLM or LLM on multimodal data
            else: # LLM or MLLM on text-only data
                if self.model_wrapper.is_overridden_generate_text_only(self.model_wrapper):
                    return self.generate_by_self('text_only', data, device)
                return self.generate_text_only(data, device)
        else:
            raise NotImplementedError(f"Unsupported type: {data['question_type']}")

    def infer_chain_of_thought(self, data, device=torch.device('cuda')):
        language = detect_language(data['prompt_instruction'])
        data['prompt_instruction'] = data['prompt_instruction'].rstrip() + ' ' + TYPE2LANGUAGE2PROMPT['cot'][language]
        return self.infer_direct(data, device)

    @abstractmethod
    def infer(self, data, device=torch.device('cuda'), **kwargs):
        raise NotImplementedError
