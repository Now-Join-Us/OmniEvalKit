import string
import numpy as np
from collections import defaultdict
from itertools import chain
from utils import flatten_list, logger

def one_hot_encode(item, length, topk=1):
    if isinstance(item, int):
        return [1 if i == item else 0 for i in range(length)]
    elif isinstance(item, list):
        if len(item) == 0:
            return [0 for i in range(length)]
        if isinstance(item[0], str):
            return [1 if string.ascii_uppercase[i] in item else 0 for i in range(length)]
        if isinstance(item[0], int):
            return [1 if i in item else 0 for i in range(length)]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def get_acc_of_multiple_choice(pred, gold):
    # Wrong choice don't get points, and the right part ones get only part of the points.
    if len(pred) != len(gold):
        raise ValueError
    correct = sum([i & j for i, j in zip(pred, gold)])
    # return 1 if correct > 0 else 0
    wrong = sum([i & ~j for i, j in zip(pred, gold)])
    # return correct / sum(gold) * 1.0 if wrong == 0 else 0.0
    correct_abs = max(0, correct - wrong)
    return correct_abs / sum(gold)

def _align_two_type(a, b):
    if type(a) == type(b):
        return a, b
    if isinstance(a, int) and isinstance(b, str):
        return string.ascii_uppercase[a], b
    elif isinstance(a, str) and isinstance(b, int):
        return a, string.ascii_uppercase[b]

class BaseCalculator:
    import evaluate as hf_evaluate
    hf_exact_match = hf_evaluate.load("evals/metrics/exact_match.py")

    @staticmethod
    def exact_match(filtered_r, gold, max_to_0_1=False, **kwargs):
        if isinstance(gold, str):
            gold = [gold]
        if isinstance(filtered_r, str):
            filtered_r = [filtered_r]

        flattened_references = flatten_list(gold)
        total_exact_match_score = sum(
            [
                BaseCalculator.hf_exact_match.compute(predictions=filtered_r, references=[i_ref] * len(filtered_r), **kwargs)['exact_match'] \
                    for i_ref in flattened_references
            ]
        )
        if max_to_0_1:
            return {'acc': 1.0} if total_exact_match_score > 0 else {'acc': 0.0}

        return {'acc': total_exact_match_score}

    @staticmethod
    def exact_in(filtered_r, gold, **kwargs):
        return {'acc': 1.0 if gold in filtered_r else 0.0}

    @staticmethod
    def multiple_choice(filtered_r, is_filtered, gold, **kwargs):
        # filtered_r: int, list of str, or list of int
        # gold: int, str, or one_hot

        if not is_filtered:
            return {'acc': 0}
        acc = 0.0
        if isinstance(gold, int) or isinstance(gold, str):
            # single choice
            if isinstance(filtered_r, list):
                if len(set(filtered_r)) == 1:
                    filtered_r = list(set(filtered_r))[0]
                    filtered_r, gold = _align_two_type(filtered_r, gold)
                    acc = 1.0 if filtered_r == gold else 0.0
                else:
                    filtered_r = filtered_r[0]
                    filtered_r, gold = _align_two_type(filtered_r, gold)
                    acc = 1.0 if filtered_r == gold else 0.0
            elif isinstance(filtered_r, int) or isinstance(filtered_r, str):
                filtered_r, gold = _align_two_type(filtered_r, gold)
                acc = 1.0 if filtered_r == gold else 0.0
            else:
                raise NotImplementedError(f'Unhandled filtered_r type: {type(filtered_r)}')
        elif isinstance(gold, list):
            # multiple choices & gold is one_hot
            filtered_r = one_hot_encode(filtered_r, length=len(gold))
            acc = get_acc_of_multiple_choice(filtered_r, gold)
        else:
            raise NotImplementedError(f'Unhandled gold type: {type(gold)}')
        return {'acc': acc}

    @staticmethod
    def loglikelihood(filtered_r, is_filtered, gold, choices_length, prompt_choices, **kwargs):
        if not is_filtered:
            return {'acc': 0, 'acc_norm': 0}
        filtered_r = filtered_r[:len(choices_length)]
        lls, is_greedy = zip(*filtered_r)
        completion_len = np.array([float(i) for i in choices_length])
        topk = sum(gold) if isinstance(gold, list) else 1
        pred = sorted(range(len(lls)), key=lambda i: lls[i], reverse=True)[:topk]
        lls_norm = lls / completion_len
        pred_norm = sorted(range(len(lls_norm)), key=lambda i: lls_norm[i], reverse=True)[:topk]

        return {
            'acc': BaseCalculator.multiple_choice(filtered_r=pred, is_filtered=True, gold=gold)['acc'],
            'acc_norm': BaseCalculator.multiple_choice(filtered_r=pred_norm, is_filtered=True, gold=gold)['acc']
        }, {
            'filtered_r': [prompt_choices[i] for i in pred],
            'filtered_r_norm': [prompt_choices[i] for i in pred_norm]
        }
